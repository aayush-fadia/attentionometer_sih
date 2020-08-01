import cv2
import pytesseract
# import uuid
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


# def get_name(frame):
#     print("Getting Name")
#     height, width = frame.shape[0], frame.shape[1]
#     frame = frame[int(0.85 * height):, :int(0.5 * width)]
#     cv2.imwrite("samples/{}.png".format(uuid.uuid4()), frame)
#     strip = preprocess_frame(frame)
#     return pytesseract.image_to_string(strip)
#
#
# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     ret, thresh2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     return thresh2
#
#
# def preproc(frames):
#     preproc_frames = []
#     for i in range(len(frames)):
#         ret, thresh2 = cv2.threshold(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY_INV)
#         preproc_frames.append(thresh2)
#     return preproc_frames


def prep2(crop):
    thresh = cv2.inRange(crop, (230, 230, 230), (255, 255, 255))
    return thresh


def thresh2(crop):
    thresh = cv2.inRange(crop, (180, 180, 180), (255, 255, 255))
    thresh = 255 - thresh
    return thresh


def crop2(bin):
    D = 10
    colmaxas = np.max(bin, 0)
    b = 0
    rightmost = True
    for i in range(len(colmaxas)):
        if colmaxas[i] != 0:
            b = 0
        else:
            b += 1
        if b > D:
            rightmost = False
            break
    r = i if rightmost else i - D + int(D / 2)
    bin = bin[:, 0:r]
    rowmaxs = np.max(bin, 1)
    start = -1
    end = -1
    for j, v in enumerate(rowmaxs):
        if v == 255 and start == -1:
            start = j
        if v == 0 and start != -1:
            end = j
            break
    return start, end, r


# def otsu():
#     img = cv2.imread('getnames.png', 0)
#     blur = cv2.GaussianBlur(img, (5, 5), 0)
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow(str(1), th3)
#     cv2.waitKey(500)


def get_name2(frame):
    height, width = frame.shape[0], frame.shape[1]
    frame = frame[int(0.85 * height):, :int(0.5 * width)]
    binary = prep2(frame)
    t, b, r = crop2(binary)
    frame = frame[t - 5:b + 5, 0:r]
    t2 = thresh2(frame)
    return pytesseract.image_to_string(t2)

# if __name__ == '__main__':
#     import os
#
#     imgs = os.listdir('samples')
#     for img in imgs:
#         img_filename = "samples/" + img
#         frame = cv2.imread(img_filename)
#
#         cv2.imshow('frame', frame)
#         cv2.imshow('prep', t2)
#         cv2.waitKey(0)
