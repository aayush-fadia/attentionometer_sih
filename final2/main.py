import cv2
from chopper import chop
from executor import process_and_upload
from concurrent.futures.thread import ThreadPoolExecutor
import numpy as np
import pyscreenshot as ImageGrab
import time

def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)

time.sleep(5)
cap = None
#cap = cv2.VideoCapture("glrec.mp4")
i = 0
is_live = True
while is_live:
    if is_live:
        frame = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(frame),  cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
    #cv2.imshow("frame", frame)
    #cv2.waitKey(1)
    if i > 10:

        imgs = chop(frame)
        with ThreadPoolExecutor() as master:
            master.map(process_and_upload, enumerate(imgs))

    i += 1
