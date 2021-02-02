import os, sys
import argparse
import cv2
import time
import dlib 
import glob 
import mxnet as mx
import numpy as np

from utils import NMS

param_receptive_field_list = [15, 20, 40, 70, 110, 250, 400, 560]
param_receptive_field_stride = [4, 4, 8, 8, 16, 32, 32, 32]
param_bbox_small_list = [10, 15, 20, 40, 70, 110, 250, 400]
param_bbox_large_list = [15, 20, 40, 70, 110, 250, 400, 560]
param_receptive_field_center_start = [3, 3, 7, 7, 15, 31, 31, 31]
param_num_output_scales = 8

class LffdDet(object):

    def __init__(self,
                 symbol_file_path,
                 model_file_path,
                 ctx=mx.cpu()):
        self.symbol_file_path = symbol_file_path
        self.model_file_path = model_file_path
        self.ctx = ctx

        self.receptive_field_list = param_receptive_field_list
        self.receptive_field_stride = param_receptive_field_stride
        self.bbox_small_list = param_bbox_small_list
        self.bbox_large_list = param_bbox_large_list
        self.receptive_field_center_start = param_receptive_field_center_start
        self.num_output_scales = param_num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]
        self.input_height = 480
        self.input_width = 640
        self.__load_model()

    def __load_model(self):
        # load symbol and parameters
        print('----> load symbol file: %s\n----> load model file: %s' % (self.symbol_file_path, self.model_file_path))
        if not os.path.exists(self.symbol_file_path):
            print('The symbol file does not exist!!!!')
            sys.exit(1)
        if not os.path.exists(self.model_file_path):
            print('The model file does not exist!!!!')
            sys.exit(1)
        self.symbol_net = mx.symbol.load(self.symbol_file_path)
        data_name = 'data'
        data_name_shape = (data_name, (1, 3, self.input_height, self.input_width))
        self.module = mx.module.Module(symbol=self.symbol_net,
                                               data_names=[data_name],
                                               label_names=None,
                                               context=self.ctx,
                                               work_load_list=None)
        self.module.bind(data_shapes=[data_name_shape],
                         for_training=False)

        save_dict = mx.nd.load(self.model_file_path)
        self.arg_name_arrays = dict()
        self.arg_name_arrays['data'] = mx.nd.zeros((1, 3, self.input_height, self.input_width), self.ctx)
        self.aux_name_arrays = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})
        self.module.init_params(arg_params=self.arg_name_arrays,
                                aux_params=self.aux_name_arrays,
                                allow_missing=True)
        print('----> Model is loaded successfully.')

    def predict(self, image, resize_scale=1, score_threshold=0.8, top_k=100, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):

        if image.ndim != 3 or image.shape[2] != 3:
            print('Only RGB images are supported.')
            return None

        bbox_collection = []

        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side

        input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

        input_image = input_image.astype(dtype=np.float32)
        input_image = input_image[:, :, :, np.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])

        data_batch = DataBatch()
        data_batch.data = [mx.ndarray.array(input_image, self.ctx)]
        
        # tic = time.time()
        self.module.forward(data_batch=data_batch, is_train=False)
        results = self.module.get_outputs()
        outputs = []
        for output in results:
            outputs.append(output.asnumpy())
        # toc = time.time()
        # infer_time = (toc - tic) * 1000

        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = np.squeeze(outputs[i * 2], (0, 1))

            # score_map_show = score_map * 255
            # score_map_show[score_map_show < 0] = 0
            # score_map_show[score_map_show > 255] = 255
            # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=np.uint8), (0, 0), fx=2, fy=2))
            # cv2.waitKey()

            bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = np.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_numpy = np.array(bbox_collection, dtype=np.float32)

        final_bboxes = NMS(bbox_collection_numpy, NMS_threshold)
        final_bboxes_ = []
        for i in range(final_bboxes.shape[0]):
            # bbox: (x1, y1, x2, y2, score, -1)
            final_bboxes_.append([final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4], -1])

        return final_bboxes_ # , infer_time



class DataBatch():
    pass 