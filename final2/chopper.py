import cv2
import numpy as np

FRAME_SIZE = (1280, 720)
nums = [cv2.resize(cv2.imread('../permutations/{}.png'.format(x), flags=cv2.IMREAD_GRAYSCALE), FRAME_SIZE) for x in
        [6, 7, 8]]


class Slicer(object):
    def __getitem__(self, idx):
        return idx


h = FRAME_SIZE[1]
w = FRAME_SIZE[0]
slices = {
    6: [
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), :int(w / 3)],
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), int(w / 3):2 * int(w / 3)],
        Slicer()[int((h / 2) - (h / 3)):int(h / 2), 2 * int(w / 3):],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)), :int(w / 3)],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 2):int((h / 2) + (h / 3)):, 2 * int(w / 3):],
    ],
    7: [
        Slicer()[:int(h / 3), :int(w / 3)],
        Slicer()[:int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[:int(h / 3), 2 * int(w / 3):],
        Slicer()[int(h / 3):2 * int(h / 3), :int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), 2 * int(w / 3):],
        Slicer()[int(2 * h / 3):, int(w / 3):2 * int(w / 3)],
    ],
    8: [
        Slicer()[:int(h / 3), :int(w / 3)],
        Slicer()[:int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[:int(h / 3), 2 * int(w / 3):],
        Slicer()[int(h / 3):2 * int(h / 3), :int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), int(w / 3):2 * int(w / 3)],
        Slicer()[int(h / 3):2 * int(h / 3), 2 * int(w / 3):],
        Slicer()[int(2 * h / 3):, int(w / 2) - (int(w / 3)):int(w / 2)],
        Slicer()[int(2 * h / 3):, int(w / 2):int(w / 2) + (int(w / 3))],
    ],
}


def show_images(imgs):
    for i, img in enumerate(imgs):
        cv2.imshow('img{}'.format(i), img)


def chop_into(frame, n):
    cur_slices = slices[n]
    rets = []
    for slc in cur_slices:
        rets.append(frame[slc].copy())
    return rets


def chop(full_frame):
    frame_color = full_frame[15:-15, 15:-15]
    frame_color = cv2.resize(frame_color, FRAME_SIZE)
    frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame, 50, 150, apertureSize=3)
    confs = [np.sum(cv2.bitwise_and(img, canny)) for img in nums]
    max_index = np.argmax(confs) + 6
    imgs = chop_into(frame_color, max_index)
    return imgs
