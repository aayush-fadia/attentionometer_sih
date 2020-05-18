import cv2
import dlib
import numpy as np
from imutils import face_utils

predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_face_keypoints(frame, face_rect):
    (t, l), (r, b) = face_rect
    dlib_rect = dlib.rectangle(l, t, b, r)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_shape = predictor(frame, dlib_rect)
    face_shape = face_utils.shape_to_np(face_shape)
    return face_shape


def draw_face_keypoints(frame, face_keypoints):
    for x, y in face_keypoints:
        cv2.circle(frame, (x, y), 1, (0, 0, 0), thickness=8)
