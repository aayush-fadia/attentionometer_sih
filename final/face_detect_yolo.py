import cv2
import numpy as np

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416
net = cv2.dnn.readNetFromDarknet('/home/aayush/SIH/attentionometer_sih/yolo_face_detect/yolov3-face.cfg',
                                 '/home/aayush/SIH/attentionometer_sih/yolo_face_detect/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layers_names = net.getLayerNames()
output_names = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    faces = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        faces.append(((top, left), (bottom, right)))
    return faces[0] if len(faces) > 0 else None


def get_face_rect_yolo(frame):
    """
    Get face rectangle using a YOLO face detector, on a CLAHE enhanced image.
    :param frame: RGB frame from the webcam
    :return: Two Tuples (points) (top, left), (bottom, right)
    """
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(output_names)
    face = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    return face


def draw_face_rect_yolo(frame, face_rect):
    if face_rect is not None:
        (t, l), (b, r) = face_rect
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255))


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        face = get_face_rect_yolo(frame)
        draw_face_rect(frame, face)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
