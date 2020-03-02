import cv2
import numpy as np
import dlib
from screeninfo import get_monitors
import random
import pickle
import os
import requests
import bz2

m = get_monitors()[0]
HEIGHT = m.height
WIDTH = m.width
img = np.zeros((m.height, m.width, 3))
DELAY = 100
SQUARE_SIZE = 30
COMPRESSED_FILENAME = 'shape_predictor_68_face_landmarks.dat.bz2'
EXTRACTED_FILENAME = 'shape_predictor_68_face_landmarks.dat'
URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


detector = dlib.get_frontal_face_detector()
ls = os.listdir('.')
if EXTRACTED_FILENAME not in ls:
    if COMPRESSED_FILENAME not in ls:
        print('Downloading File')
        myfile = requests.get(URL)
        open(COMPRESSED_FILENAME, 'wb').write(myfile.content)
    print("Extracting File")
    zipfile = bz2.BZ2File(COMPRESSED_FILENAME)
    data = zipfile.read()
    open(EXTRACTED_FILENAME, 'wb').write(data)

print('File Found.')
predictor = dlib.shape_predictor(EXTRACTED_FILENAME)
cap = cv2.VideoCapture(0)
dataset = []
while cap.isOpened():
    cv2.imshow('look here', img)
    img[:, :, :] = 255
    xc = random.randint(SQUARE_SIZE, HEIGHT - SQUARE_SIZE)
    yc = random.randint(SQUARE_SIZE, WIDTH - SQUARE_SIZE)
    img[xc - SQUARE_SIZE:xc + SQUARE_SIZE, yc - SQUARE_SIZE:yc + SQUARE_SIZE] = 0
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            (x, y, w, h) = rect_to_bb(rect)
            eyes = shape[36:48]
            shape2 = shape.copy().astype(np.float)
            print(shape2)
            for i in range(shape2.shape[0]):
                shape2[i, 0] /= image.shape[0]
                shape2[i, 1] /= image.shape[1]
            print(shape2)
            xmin = eyes[0][0]
            xmax = eyes[0][0]
            ymin = eyes[0][1]
            ymax = eyes[0][1]
            for (x, y) in shape[36:48]:
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
            xmax = xmax + 20
            ymax = ymax + 20
            xmin = xmin - 20
            ymin = ymin - 20
            # xmax = min(xmax + 10, image.shape[0])
            # ymax = min(ymax + 10, image.shape[1])
            # xmin = max(xmin - 10, 0)
            # ymin = max(ymin - 10, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0))
            nimg = image[ymin:ymax, xmin:xmax, :]
        cv2.imshow('frame', nimg)
        xcn = 2 * xc / HEIGHT - 1
        ycn = 2 * yc / WIDTH - 1
        dataset.append((nimg, xcn, ycn, shape2))
        if cv2.waitKey(DELAY) == ord('q'):
            break
    else:
        print("Face Not Found")

DATA_OUT_FOLDER = 'datasets'
if DATA_OUT_FOLDER not in ls:
    os.mkdir(DATA_OUT_FOLDER)
len = len(os.listdir(DATA_OUT_FOLDER))

with open('datasets/Dataset' + str(len) + '.pkl', 'wb') as f:
    pickle.dump(dataset, f)
