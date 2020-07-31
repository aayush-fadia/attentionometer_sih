from landmarks3d import get_face_keypoints, calculate_attention
from extract_name import get_name
from lip_var import get_lip_var2
from collections import defaultdict, deque
from multiprocessing import RLock

lock = RLock()
buffer = defaultdict(lambda: deque(maxlen=48))


def process_and_upload(frame):
    print("Processing one frame")
    name = get_name(frame)
    print("NAME IS {}".format(name))
    landmarks = get_face_keypoints(frame)
    cur_attention = calculate_attention(landmarks)
    variance, distance = get_lip_var2(buffer[name], landmarks)
    with lock:
        buffer[name].append(distance)
    print("UPLOAD {} FOR {}".format(cur_attention, name))
