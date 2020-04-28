import cv2

<<<<<<< HEAD
from face_detect_dlib import get_face_rect_dlib
from face_keypoint_predict_dlib import get_face_keypoints, draw_face_keypoints

=======
from collections import deque
from face_detect_dlib import get_face_rect_dlib
from face_keypoint_predict_dlib import get_face_keypoints, draw_face_keypoints

FILL_MEMORY = False
EVERY_N_FRAMES = 20
NUM_FRAMES_PASSED = 0
NUM_FRAMES_COLLECTED = 0
TOTAL_FRAMES = 48
CONSEC_FRAMES = deque()
is_eye_closed = 0
is_eye_closed_weight = 1.0
lip_variance = 0.0
lip_variance_weight = 1.0
eye_gaze_variance = 0.0
eye_gaze_variance_weight = 1.0

>>>>>>> 3790dead6bf0551b595a333d9e0039e0c8a0fc8a

def show_output(frame):
    cv2.imshow('final', frame)
    cv2.waitKey(1)


<<<<<<< HEAD
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
=======
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
>>>>>>> 3790dead6bf0551b595a333d9e0039e0c8a0fc8a
    face_rect = get_face_rect_dlib(frame)
    if face_rect is None:
        print("No Face Found!")
        show_output(frame)
        continue
    face_keypoints = get_face_keypoints(frame, face_rect)
    draw_face_keypoints(frame, face_keypoints)
<<<<<<< HEAD
    show_output(frame)
=======
    NUM_FRAMES_PASSED += 1
    if NUM_FRAMES_PASSED == EVERY_N_FRAMES:
        ## eye_gaze_variance = CALL EYE GAZE VARIANCE ##
        NUM_FRAMES_PASSED = 0

    if FILL_MEMORY is False:
        CONSEC_FRAMES.append(face_keypoints)
        NUM_FRAMES_COLLECTED += 1
        if NUM_FRAMES_COLLECTED == TOTAL_FRAMES:
            FILL_MEMORY = True
    else:
        ## is_eye_closed = CALL EYE CLOSURE (list(CONSEC_FRAMES)) ##
        ## lip_variance = CALL LIP VARIANCE (list(CONSEC_FRAMES)) ##
        CONSEC_FRAMES.popleft()
        CONSEC_FRAMES.append(face_keypoints)

    show_output(frame)
>>>>>>> 3790dead6bf0551b595a333d9e0039e0c8a0fc8a
