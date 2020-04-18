from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Start=0
End=27
cap=cv2.VideoCapture(0)
print(cap.isOpened())
while(cap.isOpened()):
    ret,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
       # landmarks=predictor(gray,face)
        #for n in range(0,26):
            #x=landmarks.part(n).x
            #y=landmarks.part(n).y
            #cv2.circle(frame, (x, y),2, (0, 255, 0), -1)
       shape = predictor(gray, face)
       shape = face_utils.shape_to_np(shape)
       FaceBoundary = shape[Start:End]
       FaceHull = cv2.convexHull(FaceBoundary)
       cv2.drawContours(frame, [FaceHull], -1, (0, 255, 0), 1)
    cv2.imshow('frames',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()

cv2.destroyAllWindows()