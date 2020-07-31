import cv2
from SIH_Modified_Segmentation import get_frames
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def get_name(frame):
    frame = preprocess_frame(frame)
    height, width = frame.shape[0],frame.shape[1]
    strip = frame[int(0.9*height):,:int(0.5*width)]
    return pytesseract.image_to_string(strip)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh2 = cv2.threshold(gray,227,255,cv2.THRESH_BINARY_INV)
    return thresh2

def preproc(frames):
    preproc_frames=[]
    for i in range(len(frames)):
        ret,thresh2 = cv2.threshold(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),230,255,cv2.THRESH_BINARY_INV)
        preproc_frames.append(thresh2)
    return preproc_frames

def test():
    img = cv2.imread('getnames.png',1)
    frames = get_frames(img,4)
    names=[]
    frames = preproc(frames)
    
    for i in range(len(frames)):
        cv2.imshow(str(i),frames[i])
    cv2.waitKey(10000)
    
    for frame in frames:
        names.append(get_name(frame))
        

def otsu():
    img = cv2.imread('getnames.png',0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow(str(1),th3)
    cv2.waitKey(500)

test()
