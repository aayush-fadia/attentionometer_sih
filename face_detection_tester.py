import cv2
import dlib
import numpy as np


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def prep_gray(grayold):
    return clahe.apply(grayold)


def get_eye_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = prep_gray(gray)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for rect in rects:
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            for x, y in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    key = cv2.waitKey(1) & 0xFF
    return key


cap = cv2.VideoCapture(0)
key = -1
while not key == ord('q'):
    ret, frame = cap.read()
    key = get_eye_region(frame)
cv2.destroyAllWindows()
