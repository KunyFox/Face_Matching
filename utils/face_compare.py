import cv2, os
import mxnet as mx 
import numpy as np 


def cosine(f1, f2):
    s = np.sum(np.multiply(f1, f2), axis=1) / \
        (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1))
    return s

def pearson(f1, f2):
    f1 = f1 - np.mean(f1, axis=1)[:, None]
    f2 = f2 - np.mean(f2, axis=1)[:, None]
    s = np.sum(np.multiply(f1, f2), axis=1) / \
        (np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1))
    # (-1, 1) -> (0, 1)
    return 0.5 + 0.5 * s 


func_dict = {
    'cosine': cosine,
    'pearson': pearson
}


class FaceCompare(object):
    def __init__(self, 
                 lib_path,
                 face_encoder,
                 thresh,
                 func='pearson',
                 ctx=mx.cpu()):
        self.lib_path = lib_path
        self.face_encoder = face_encoder
        self.ctx = ctx 
        self.thresh = thresh
        self._get_face_encoding()
        if func not in list(func_dict.keys()):
            func = 'pearson' 
        self.simi_func = func_dict[func]

    def compare(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        feat_encoding = self.face_encoder.predict(x)
        score, indx = self.calculate_similarity(feat_encoding)
        if score <= self.thresh:
            indx = -1
        else:
            indx = self.face_ind2id[indx]
        return indx, score

    def calculate_similarity(self, feat):
        assert feat.shape[0] == 1
        similarity = self.simi_func(feat, self.face_encoding)
        # similarity = np.sum(np.multiply(feat, self.face_encoding), axis=1) / \
        #         (np.linalg.norm(feat, axis=1) * np.linalg.norm(self.face_encoding, axis=1))
        max_index = np.argmax(similarity)
        max_score = similarity[max_index]
        return max_score, max_index
        
        
    
    def _get_face_encoding(self):
        assert os.path.isdir(self.lib_path), "{} is not face-lib dir.".format(self.lib_path)
        folds = os.listdir(self.lib_path)
        face_id = [f for f in folds if f[0] is not '.']
        assert len(face_id) > 0, "There is no face under {}.".format(self.lib_path)
        face_ind2id = []
        face_ecoding = [] 
        for ith, id in enumerate(face_id):
            id_root = os.path.join(self.lib_path, id)
            imgs = os.listdir(id_root)
            imgs = [im for im in imgs if im.endswith('.jpg')]
            for img in imgs:
                pic = cv2.cvtColor(cv2.imread(os.path.join(id_root, img)), cv2.COLOR_BGR2RGB)
                pic = cv2.resize(pic, (112, 112))
                face_ecoding.append(self.face_encoder.predict(pic))
                face_ind2id.append(ith)
        face_ecoding = np.concatenate(face_ecoding, axis=0)
        face_id.append("UNKNOWN")
        self.face_id = face_id
        self.face_ind2id = face_ind2id
        self.face_encoding = face_ecoding