import threading
#from Database import DataBase
from extract_name import get_name2
from landmarks3d import get_face_keypoints, calculate_attention, calculate_vector
from lip_var import get_lip_dist, get_lip_variance
from nodding import isNodding
from yawn import isYawn

#db = DataBase("Teacher")


def process_and_upload(frame, buffer):
    name = get_name2(frame)
    buffer.announce(name)
    landmarks = get_face_keypoints(frame)
    if landmarks is not None:
        # Primary Calculations
        landmarks = landmarks[0]
        cur_vect = calculate_vector(landmarks)
        cur_attention = calculate_attention(cur_vect)
        dist = get_lip_dist(landmarks)
        yawn = isYawn(landmarks)
        # Buffer Primary Values
        buffer.add_lip_dist(name, dist)
        buffer.add_orientation_vector(name, cur_vect)
        buffer.add_attention_score(name, cur_attention)
        buffer.add_yawn(name, yawn)
        # Secondary Calculations
        variance = get_lip_variance(buffer.lip_distances[name])
        nod = isNodding(buffer.orientation_vectors[name])
        # Buffer Secondary Values
        buffer.add_variance(name, variance)
        buffer.add_nod(name, nod)
        #t1 = threading.Thread(target=db.insert_data, args=(name[0:-1], cur_attention))
        #t1.start()
        #print("Uploading {}attention for {}".format(cur_attention, name))
