# mouth -> (48, 68)
# inner_mouth -> (60, 68)
# right_eyebrow -> (17, 22)
# left_eyebrow -> (22, 27)
# right_eye -> (36, 42)
# left_eye -> (42, 48)
# nose -> (27, 36)
# jaw -> (0, 17)
from scipy.spatial import distance as dist
import numpy as np
from _collections import deque


def get_tot_dist(mouth):
    mean_pos = np.sum(mouth, axis=0) / 12
    upper_lip_dist = dist.euclidean(mean_pos, mouth[3])
    lower_lip_dist = dist.euclidean(mean_pos, mouth[9])
    return upper_lip_dist + lower_lip_dist


def get_lip_var(all_frames):
    distances = deque(maxlen=48)
    (start, end) = 48, 60  # (48,60) -> outer mouth, (60,68) -> inner mouth
    for shape in all_frames:
        mouth = shape[start:end]
        total_distance = get_tot_dist(mouth)
        distances.append(total_distance)
    variance = np.var(distances)
    print("variance:", variance)
    return variance


def get_lip_var2(distances, current_keypoints):
    mouth = current_keypoints[48:60]
    total_distance = get_tot_dist(mouth)
    if len(distances) == 0:
        return -1, total_distance
    variance = np.var(distances + [total_distance])
    return variance, total_distance
