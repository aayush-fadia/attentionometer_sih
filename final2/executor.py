from landmarks3d import get_face_keypoints, calculate_attention
from extract_name import get_name2
from lip_var import get_lip_var2
from collections import defaultdict, deque
from multiprocessing import RLock
from Database import DataBase
import threading

lock = RLock()
buffer = defaultdict(lambda: deque(maxlen=48))
db = DataBase("Teacher")
oldi = -1
oldname = ""

def process_and_upload(enumframes):
    i, frame = enumframes

    name = get_name2(frame)

    landmarks = get_face_keypoints(frame)
    if landmarks is not None:
        cur_attention = calculate_attention(landmarks)
        variance, distance = get_lip_var2(buffer[name], landmarks)
        if distance != -1:
            with lock:
                buffer[name].append(distance)
        if variance != -1:
            cur_attention += (0.001 * variance)
        t1 = threading.Thread(target=db.insert_data, args=(name[0:-1], cur_attention))
        t1.start()
        print("Uploading {}attention for {}".format(cur_attention, name))