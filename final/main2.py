import cv2
from face_align import get_face_keypoints

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    kps = get_face_keypoints(frame)
    if kps is not None:
        kps = kps[0]
        for keypoint in kps:
            print(keypoint)
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
    cv2.imshow('face', frame)
    cv2.waitKey(1)
