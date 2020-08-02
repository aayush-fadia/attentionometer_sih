#from face_detect_dlib import get_face_rect_dlib
#from face_keypoint_predict_dlib import get_face_keypoints, draw_face_keypoints
#import cv2
#import face_alignment
import numpy as np
from scipy.spatial import distance as dist

def isYawn(kps):
    mouth = kps[48:60]
    left = mouth[0]
    right = mouth[6]
    top = mouth[3]
    bottom = mouth[9]
    hor = dist.euclidean(left, right)
    ver = dist.euclidean(top, bottom)
    ratio = ver/hor
    if ratio>=0.9:
        return 1
    else:
        return 0

def show_output(frame):
    cv2.imshow('final', frame)
    cv2.waitKey(1)

def get_face_keypoints(frame):
    return fa.get_landmarks(frame)

def test():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        #face_rect = get_face_rect_dlib(frame)
        face_rect = get_face_keypoints(frame)
        if face_rect is None:
            print("No Face Found!")
            show_output(frame)
            continue
        #face_keypoints = get_face_keypoints(frame, face_rect)
        face_keypoints = face_rect[0]
        yawn = isYawn(face_keypoints)
        if yawn==1:
            print("he is yawning!")
        mouth = face_keypoints[48:60]
        cv2.circle(frame, (int(mouth[0][0]), int(mouth[0][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[6][0]), int(mouth[6][1])), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(mouth[3][0]), int(mouth[3][1])), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(mouth[9][0]), int(mouth[9][1])), 3, (0, 0, 255), -1)
        #for keypoint in mouth:
        #    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), 3, (255, 0, 0), -1)
        #mhull = cv2.convexHull(mouth)
        #cv2.drawContours(frame, mhull, -1, (0, 255, 0), 1)
        show_output(frame)