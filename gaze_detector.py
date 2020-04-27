import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # or even "-1"

import cv2
import dlib
from screeninfo import get_monitors
import numpy as np
from tensorflow.keras.models import load_model

m = get_monitors()[0]
HEIGHT = m.height
WIDTH = m.width
VIDEO_SOURCE = 0


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/aayush/SIH/attentionometer_sih/shape_predictor_68_face_landmarks.dat")
lc_model = load_model("/home/aayush/SIH/mpii-gaze/saved_models/fconv2.ADO.lc.12:24Apr19/vacc_best.h5")
rc_model = load_model("/home/aayush/SIH/mpii-gaze/saved_models/fconv2.ADO.rc.14:49Apr20/vacc_best.h5")
EYE = 'left'


def screen_image_col(col_index):
    img = np.zeros((HEIGHT, WIDTH, 3))
    img[:, :, :] = 255
    l = int(WIDTH * col_index / 6)
    r = int(WIDTH * (col_index + 1) / 6)
    img[:, l:r, :] = 0
    return img


def get_eye_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            eyes = shape[36:48]
            shape2 = shape.copy().astype(np.float)
            for i in range(shape2.shape[0]):
                shape2[i, 0] /= frame.shape[0]
                shape2[i, 1] /= frame.shape[1]
            xmin = np.min(eyes[:, 0])
            xmax = np.max(eyes[:, 0])
            ymin = np.min(eyes[:, 1])
            ymax = np.max(eyes[:, 1])
            return xmin, xmax, ymin, ymax
    else:
        print("Face Not Found")
        return None, None, None, None


def get_eye(eyes_image):
    shape = eyes_image.shape
    if EYE == 'left':
        eye_image = eyes_image[:, :int(shape[1] / 2), :]
        eye_image = cv2.resize(eye_image, (72, 40))[:, :, ::-1].astype(np.float32) / 255.
        return eye_image


def prep_eye_image(eye_image):
    eye_image = cv2.resize(eye_image, (72, 40))[:, :, ::-1].astype(np.float32) / 255.
    return eye_image


def predict(image):
    left_eye = prep_eye_image(eyes_image[:, :int(image.shape[1] / 2), :])
    right_eye = prep_eye_image(eyes_image[:, int(image.shape[1] / 2):, :])
    probs_lc = lc_model.predict(np.asarray([left_eye]))[0]
    probs_rc = lc_model.predict(np.asarray([right_eye]))[0]
    return np.argmax(probs_lc + probs_rc)


cap = cv2.VideoCapture(0)
cv2.namedWindow("looking here", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("looking here", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while 1:
    ret, frame = cap.read()
    l, r, t, b = get_eye_region(frame)
    if l is not None:
        eyes_image = frame[t - 20:b + 10, l - 20:r + 20, :]
        cv2.imshow('eye_image', eyes_image)
        index = predict(eyes_image)
        cv2.imshow('looking here', screen_image_col(index))
        cv2.waitKey(1)
