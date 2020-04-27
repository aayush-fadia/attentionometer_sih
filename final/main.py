import cv2

from face_detect_dlib import get_face_rect_dlib
from face_keypoint_predict_dlib import get_face_keypoints, draw_face_keypoints


def show_output(frame):
    cv2.imshow('final', frame)
    cv2.waitKey(1)


cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    face_rect = get_face_rect_dlib(frame)
    if face_rect is None:
        print("No Face Found!")
        show_output(frame)
        continue
    face_keypoints = get_face_keypoints(frame, face_rect)
    draw_face_keypoints(frame, face_keypoints)
    show_output(frame)