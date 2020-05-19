from collections import deque

import cv2

from New_eye_closure import Is_eye_closed
from face_detect_dlib import get_face_rect_dlib
from face_detect_yolo import get_face_rect_yolo
from face_keypoint_predict_dlib import get_face_keypoints, draw_face_keypoints
from gaze_detector import predict_column
from lip_var import get_lip_var

EVERY_N_FRAMES = 20
NUM_FRAMES_PASSED = 0
TOTAL_FRAMES = 48
frame_buffer = deque(maxlen=TOTAL_FRAMES)
is_eye_closed = 0
is_eye_closed_weight = 1.0
lip_variance = 0.0
lip_variance_weight = 1.0
eye_gaze_variance = 0.0
eye_gaze_variance_weight = 1.0


def show_output(frame):
    cv2.imshow('final', frame)
    cv2.waitKey(1)


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    face_rect = get_face_rect_yolo(frame)
    if face_rect is None:
        print("No Face Found!")
        show_output(frame)
        continue
    face_keypoints = get_face_keypoints(frame, face_rect)
    NUM_FRAMES_PASSED += 1
    if NUM_FRAMES_PASSED == EVERY_N_FRAMES:
        ## eye_gaze_variance = CALL EYE GAZE VARIANCE ##
        NUM_FRAMES_PASSED = 0
    frame_buffer.append(face_keypoints)
    if len(frame_buffer) == TOTAL_FRAMES:
        is_eye_closed = Is_eye_closed(frame_buffer)
        lip_variance = get_lip_var(frame_buffer)
    gaze_column = predict_column(frame, face_keypoints)
    draw_face_keypoints(frame, face_keypoints)
    print(gaze_column)
    show_output(frame)
