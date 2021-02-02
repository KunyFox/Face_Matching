import cv2 
import sys 
import dlib 
import numpy as np 
import mxnet as mx 

sys.path.append("..")

def xyxy2xywh(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2] - x 
    h = bbox[3] - y 
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
	    coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def landmark_predictor(path='model_zoo/shape_predictor_68_face_landmarks.dat'):
    return dlib.shape_predictor(path)

def predict_face_landmark(img, predictor, bbox):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmark = predictor(img, dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
    landmark = shape_to_np(landmark)
    # right eye: 36:42
    # left eye: 42:48
    return landmark

def vis_landmarks(img, landmark):
    for (x, y) in landmark:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
    return img

def face_align(img, left_eye, right_eye, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
    center_l = left_eye.mean(axis=0).astype('int')
    center_r = right_eye.mean(axis=0).astype('int')
    dy = center_r[1] - center_l[1]
    dx = center_r[0] - center_l[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth
    eyesCenter = ((center_l[0] + center_r[0]) // 2, (center_l[1] + center_r[1]) // 2)
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])
    aligned_img = cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)
    return aligned_img


class FaceAligner(object):
    def __init__(self, 
                 desiredLeftEye=(0.32, 0.32), 
                 desiredFaceWidth=112, 
                 desiredFaceHeight=112):
        self.desiredLeftEye = desiredLeftEye 
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        self.lm_predictor = landmark_predictor()

    def align(self, img, bbox):
        landmark = predict_face_landmark(img, self.lm_predictor, bbox)
        img = face_align(img, landmark[42:48], landmark[36:42], desiredLeftEye=self.desiredLeftEye, \
              desiredFaceWidth=self.desiredFaceWidth, desiredFaceHeight=self.desiredFaceHeight)
        return img 

'''
def main():
    img = cv2.imread('/Users/kunyuan/Desktop/face_recognization/tem.jpg')
    predictor = landmark_predictor()
    # predict face landmarks
    landmark = predict_face_landmark(img, predictor)
    # landmark visualizations
    vis_landmarks(img, './tem_marked.jpg', landmark)
    # face alignment
    aligned_img = face_align(img, landmark[42:48], landmark[36:42])
    cv2.imwrite('tem_aligned.jpg', aligned_img)

main()
'''