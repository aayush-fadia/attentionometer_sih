import cv2
import dlib

detector = dlib.get_frontal_face_detector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def get_face_rect_dlib(frame):
    """
    Get face rectangle using DLib's face detector, on a CLAHE enhanced image.
    :param frame: RGB frame from the webcam
    :return: Two Tuples (points) (top, left), (bottom, right)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    rects = detector(gray, 0)
    return ((rects[0].top(), rects[0].left()), (rects[0].bottom(), rects[0].right())) if len(rects) > 0 else None


def draw_face_rect(frame, face_rect):
    if face_rect is not None:
        (t, l), (b, r) = face_rect
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 0))


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        rectpts = get_face_rect_dlib(frame)
        draw_face_rect(frame, rectpts)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
