from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat

import cv2

from buffers import Buffer
from chopper import chop
from classifier import classify
from executor import process_and_upload


def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)


buffer = Buffer()
cap = cv2.VideoCapture("../FinalCut.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    imgs = chop(frame)
    buffer.reset_people()
    with ThreadPoolExecutor() as master:
        master.map(process_and_upload, imgs, repeat(buffer))
    classes = classify(buffer)
    buffer.set_pressences()
