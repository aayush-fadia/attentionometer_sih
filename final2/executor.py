from landmarks3d import get_face_keypoints, calculate_attention
#from extract_name import get_name
from lip_var import get_lip_var2
from collections import defaultdict, deque
from multiprocessing import RLock
from Database import DataBase

lock = RLock()
buffer = defaultdict(lambda: deque(maxlen=48))
db = DataBase("Teacher")

def process_and_upload(enumframes):
    i, frame = enumframes
    print("Processing one frame")
    #name = get_name(frame)
    name = str(i)
    print("NAME IS {}".format(name))
    landmarks = get_face_keypoints(frame)
    cur_attention = calculate_attention(landmarks)
    variance, distance = get_lip_var2(buffer[name], landmarks)
    #print("Attention", cur_attention)
    #print("lip var", 0.001*variance)
    if distance != -1:
        with lock:
            buffer[name].append(distance)
    if variance!=-1:
        cur_attention+=(0.001*variance)

    #print("UPLOAD {} FOR {}".format(cur_attention, name))
    db.insert_data(name, cur_attention)