import cv2
import numpy as np
import dlib
from screeninfo import get_monitors
import random
import pickle
import os
import requests
import bz2
import random
import string

m = get_monitors()[0]
HEIGHT = m.height
WIDTH = m.width
DELAY = 100
SQUARE_SIZE = 30
COMPRESSED_FILENAME = 'shape_predictor_68_face_landmarks.dat.bz2'
EXTRACTED_FILENAME = 'shape_predictor_68_face_landmarks.dat'
URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
VIDEO_SOURCE = 0


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


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
dataset = []


def get_random_square_image():
    img = np.zeros((m.height, m.width, 3))
    img[:, :, :] = 255
    xc = random.randint(SQUARE_SIZE, HEIGHT - SQUARE_SIZE)
    yc = random.randint(SQUARE_SIZE, WIDTH - SQUARE_SIZE)
    img[xc - SQUARE_SIZE:xc + SQUARE_SIZE, yc - SQUARE_SIZE:yc + SQUARE_SIZE] = 0
    return img, xc, yc


def get_eye_region(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            eyes = shape[36:48]
            shape2 = shape.copy().astype(np.float)
            for i in range(shape2.shape[0]):
                shape2[i, 0] /= frame.shape[0]
                shape2[i, 1] /= frame.shape[1]
            xmin = np.min(eyes[:, 0]) - 20
            xmax = np.max(eyes[:, 0]) + 20
            ymin = np.min(eyes[:, 1]) - 20
            ymax = np.max(eyes[:, 1]) + 20
            nimg = frame[ymin:ymax, xmin:xmax, :]
            return nimg, shape2
    else:
        print("Face Not Found")


while True:
    square_image, xc, yc = get_random_square_image()
    cv2.imshow('look here', square_image)
    ret = cv2.waitKey(0)
    if ret == ord(' '):
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        ret, frame = cap.read()
        eye_region, shape = get_eye_region(frame)
        cv2.imshow('frame', eye_region)
        xcn = 2 * xc / HEIGHT - 1
        ycn = 2 * yc / WIDTH - 1
        dataset.append((eye_region, xcn, ycn, shape))
        del cap
    elif ret == ord('q'):
        break
    elif ret == ord('c'):
        continue

DATA_OUT_FOLDER = 'datasets'
if DATA_OUT_FOLDER not in ls:
    os.mkdir(DATA_OUT_FOLDER)
len_files = len(os.listdir(DATA_OUT_FOLDER))
random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
with open('datasets/ds' + str(len_files) + random_string + '.pkl', 'wb') as f:
    pickle.dump(dataset, f)
