import os, sys
import argparse
import cv2
import time
import dlib 
import glob 
import mxnet as mx
import numpy as np

from mxnet.gluon.data.vision.transforms import ToTensor


HEIGTH = 112
WIDTH = 112



class FeatureEncoder(object):
    def __init__(self, 
                 symbol_path,
                 params_path, 
                 ctx=mx.cpu()):
        self.symbol_path = symbol_path
        self.params_path = params_path 
        self.ctx = ctx
        self._build_model()


    def _build_model(self):
        assert os.path.isfile(self.symbol_path), '{} is not a symbol file'.format(self.symbol_path)
        assert os.path.isfile(self.params_path), '{} is not a parameters file'.format(self.params_path)
        symbol = mx.sym.load(self.symbol_path)
        self.model = mx.gluon.nn.SymbolBlock(outputs=symbol, inputs=mx.sym.var('data'))
        self.model.load_parameters(self.params_path, ctx=self.ctx)

    def predict(self, x):
        assert x.shape[:2] == (HEIGTH, WIDTH), 'invalid shape of input'
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            x = x[:,:,:,np.newaxis]
            x = x.transpose([3, 2, 0, 1])
        x = mx.nd.array(x, self.ctx) # / 255.
        output = self.model(x)
        output = mx.nd.L2Normalization(output, mode='instance')
        output = output.asnumpy()
        return output 