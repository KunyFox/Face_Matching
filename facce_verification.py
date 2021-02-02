import os, sys
import argparse
import cv2
import time
import dlib 
import glob 
import mxnet as mx
import numpy as np

sys.path.append("..")

from models import LffdDet, FeatureEncoder
from utils import FaceAligner, enlarge_bboxes, FaceCompare, face_track

det_sym_path = "model_zoo/detector/lffd_facedet.json"
det_para_path = "model_zoo/detector/lffd_facedet.params"
extor_sym_path = "model_zoo/extractor/mobilefacenet-symbol-1111.json"
extor_para_path = "model_zoo/extractor/mobilefacenet-1111.params"

HEIGHT = 1080
WIDTH = 1920
SCALE = 8
FACE_LIB = './face_lib'
IOU_THRES = 0.7
SCORE_THRES = 0.98
FACE_TRACK = True
AREA_THRES = 900


def parse_args():
    parser = argparse.ArgumentParser(description="Demo of face detection and verification")
    parser.add_argument('--context', type=str, default='cpu',
                        help='The context (cpu or gpu) of model,')
    parser.add_argument('--face-lib', type=str, default=None,
                        help='The face lib for verifivation.')
    parser.add_argument('--shape', type=tuple, default=(1080, 1920),
                        help='The shape of video/image.') 
    parser.add_argument('--scale', type=int, default=8,
                        help='The scale for image resize.') 
    parser.add_argument('--HEIGTH', type=int, default=1080,
                        help='The IOU for NMS in face tracking.')
    args = parser.parse_args()
    return args 


def build_model(ctx):
    assert os.path.isfile(det_sym_path), "{} is not a file.".format(det_sym_path)
    assert os.path.isfile(det_para_path), "{} is not a file.".format(det_para_path)
    assert os.path.isfile(extor_sym_path), "{} is not a file.".format(extor_sym_path)
    assert os.path.isfile(extor_para_path), "{} is not a file.".format(extor_para_path)

    face_detector = LffdDet(symbol_file_path=det_sym_path,
                            model_file_path=det_para_path,
                            ctx=ctx)
    face_encoder = FeatureEncoder(symbol_path=extor_sym_path,
                                  params_path=extor_para_path,
                                  ctx=ctx)
    face_compare = FaceCompare(lib_path=FACE_LIB, 
                               face_encoder=face_encoder,
                               thresh=SCORE_THRES)
    face_aligner = FaceAligner()
    return face_detector, face_compare, face_aligner

def area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def get_face_lib(lib_path, feats_extector):
    assert os.path.isdir(lib_path), "{} is not face-lib dir.".format(lib_path)
    folds = os.listdir(lib_path)
    face_id = [f for f in folds if f[0] is not '.']
    assert len(face_id) > 0, "There is no face under {}.".format(lib_path)
    face_ind2id = []
    face_ecoding = [] 
    for id in face_id:
        id_root = os.path.join(lib_path, id)
        imgs = os.listdir(id, root)
        imgs = [im for im in imgs if im.endswith('.jpg')]
        for img in imgs:
            pic = cv2.cvtColor(cv2.imread(os.path.join(id_root, img)), cv2.COLOR_BGR2RGB)
            face_ecoding.append(feats_extector(pic))
            face_ind2id.append(id)
    face_ecoding = np.concatenate(face_ecoding, axis=0)

    return face_ecoding, face_ind2id



def main():
    face_detector, face_compare, face_aligner = build_model(mx.gpu())
    # build cam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    pre_bboxes = []
    pre_result = []
    while(cap.isOpened()):
        ret, flame = cap.read()
        if ret  == True:
            flame = cv2.flip(flame, 1)
            h, w, _ = flame.shape
            flame_ = cv2.resize(flame,(w//SCALE, h//SCALE))
            bboxes = face_detector.predict(flame_, resize_scale=1, score_threshold=0.9, top_k=10000, \
                                                        NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

            if len(pre_bboxes) > 0:
                pre_frame_bboxes, pre_frame_result, unknown_bboxes = face_track(pre_bboxes, pre_result, bboxes)
            else:
                pre_frame_bboxes = []
                pre_frame_result = []
                unknown_bboxes = bboxes
            #to_encoding_bboxes = [enlarge_bboxes(bbox, flame_.shape) for bbox in unknown_bboxes]
            to_encoding_bboxes = unknown_bboxes
            to_encoding_faces = [face_aligner.align(flame_, bbox) for bbox in to_encoding_bboxes]
            # result = []
            final_bboxes = pre_frame_bboxes
            final_result = pre_frame_result
            for face, bbox in zip(to_encoding_faces, unknown_bboxes):
                if area(bbox) < AREA_THRES:
                    indx = -1
                    score = 0.0
                else:
                    indx, score = face_compare.compare(face)
                    if score < SCORE_THRES:
                        indx = -1
                bbox[-1] = indx
                bbox[-2] = score
                final_bboxes.append(bbox)
                final_result.append(face_compare.face_id[indx])

            # final_bboxes = pre_frame_bboxes + unknown_bboxes
            # final_result = pre_frame_result + result
            if FACE_TRACK:
                pre_bboxes = final_bboxes
                pre_result = final_result
            else:
                pre_bboxes = []
                pre_result = []

            for bbox, name in zip(final_bboxes, final_result):
                score = bbox[-2]
                bbox = [int(bb*SCALE+SCALE//2) for bb in bbox]
                if name == 'UNKNOWN':
                    color = (0, 0, 255)
                    cv2.rectangle(flame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                else:
                    color = (255, 255, 0)
                    cv2.rectangle(flame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                    cv2.rectangle(flame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), color, -1)
                    cv2.putText(flame, ' '+name+'  {:.2f}'.format(score), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow('video', flame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
        else:
            break


            
if __name__ == "__main__":
    main()
