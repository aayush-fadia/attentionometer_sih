import os

print(os.getcwd())

import cv2
from chopper import chop
from executor import process_and_upload
from concurrent.futures.thread import ThreadPoolExecutor


def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)


cap = cv2.VideoCapture("glrec.mp4")
i=0
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    if i>3000:
        imgs = chop(frame)
        with ThreadPoolExecutor() as master:
            master.map(process_and_upload, enumerate(imgs))
    i+=1