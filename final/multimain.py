from collections import deque
import concurrent.futures as cf
import cv2

from New_eye_closure import Is_eye_closed
# from face_detect_dlib import get_face_rect_dlib
from face_keypoint_predict_dlib import draw_face_keypoints
# from gaze_detector import predict_column
from lip_var import get_lip_var
from face_align import get_face_keypoints

EVERY_N_FRAMES = 20
NUM_FRAMES_PASSED = 0

frame_buffers = []

is_eye_closed_weight = 1.0

lip_variance_weight = 1.0
eye_gaze_variance = 0.0
eye_gaze_variance_weight = 1.0
delay = 0


def show_output(frame, frame_id):
    cv2.imshow(str(frame_id), frame)
    cv2.waitKey(1)


def get_attention(frame, pid, frame_buffer):
    is_eye_closed = 0
    lip_variance = 0.0
    TOTAL_FRAMES = 48
    # face_rect = get_face_rect_dlib(frame)
    kps = get_face_keypoints(frame)
    if kps is None:
        print("No Face Found!")
        return (50, frame, frame_buffer)
    face_keypoints = kps[0]
    frame_buffer.append(face_keypoints)
    # print("fblen:",len(frame_buffer))
    if len(frame_buffer) == TOTAL_FRAMES:
        is_eye_closed = Is_eye_closed(frame_buffer)
        lip_variance = get_lip_var(frame_buffer)
    # gaze_column = predict_column(frame, face_keypoints)
    draw_face_keypoints(frame, face_keypoints)
    # print(gaze_column)
    return (100 - 100 * lip_variance, frame, frame_buffer)


def init_buffer():
    for i in range(10):
        frame_buffers.append(deque(maxlen=48))


def get_all_frames(frame):
    height, width = frame.shape[:2]
    # 1st quadrant
    start_row, start_col = int(0), int(0)
    end_row, end_col = int(height * .5), int(width * .5)
    qd1 = frame[start_row:end_row, start_col:end_col]
    # 2nd quadrant
    start_row, start_col = int(0), int(width * .5)
    end_row, end_col = int(height * .5), int(width)
    qd2 = frame[start_row:end_row, start_col:end_col]

    # 3rd Quadrant
    start_row, start_col = int(height * .5), int(0)
    end_row, end_col = int(height), int(width * .5)
    qd3 = frame[start_row:end_row, start_col:end_col]

    # 4th Quadrant
    start_row, start_col = int(height * .5), int(width * .5)
    end_row, end_col = int(height), int(width)
    qd4 = frame[start_row:end_row, start_col:end_col]

    return [qd1, qd2, qd3, qd4]


init_buffer()
gap = 1
cap = cv2.VideoCapture("zoom_1.mp4")
while cap.isOpened():
    ret, frames = cap.read()
    frame_list = get_all_frames(frames)
    gap += 1
    if (gap % 10 == 0):
        with cf.ThreadPoolExecutor() as executor:
            att_frames = [executor.submit(get_attention, frame, i, frame_buffers[i]).result() for i, frame in
                          enumerate(frame_list)]

        for k in range(4):
            frame_buffers[k] = att_frames[k][2]
            show_output(att_frames[k][1], k)

        print(chr(27) + '[2j')
        print('\033c')
        print('\x1bc')
        print("attention scores\n0: ", att_frames[0][0], "\n1: ", att_frames[1][0], "\n2: ", att_frames[2][0], "\n3: ",
              att_frames[3][0], "\n\n\n")
    else:
        for x, final in enumerate(frame_list):
            show_output(final, x)
