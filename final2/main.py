from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat
from threading import Thread

import cv2

from buffers import Buffer
from chopper import chop
from classifier import classify
from executor import process_and_upload


def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)


buffer = Buffer()
frame = None
cap = cv2.VideoCapture("../FinalCut.mp4")


def process_frame():
    imgs = chop(frame)
    buffer.reset_people()
    with ThreadPoolExecutor() as master:
        master.map(process_and_upload, imgs, repeat(buffer))
    classes = classify(buffer)
    buffer.set_pressences()


processingThread = Thread(target=process_frame)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if not processingThread.is_alive():
        #cv2.imshow('processing', frame)
        processingThread = Thread(target=process_frame)
        processingThread.start()
