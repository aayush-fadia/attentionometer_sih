import cv2
import dlib

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    cv2.imshow('frame', img)
    cv2.waitKey(1)
