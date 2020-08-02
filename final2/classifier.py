from collections import defaultdict
from enum import Enum

import numpy as np


class AttentionClass(Enum):
    UNKNOWN = 0
    DROWSY = 1
    INATTENTIVE = 2
    ATTENTIVE = 3
    INTERACTIVE = 4


def classify(buffer):
    classes = defaultdict(lambda: AttentionClass.UNKNOWN)
    scores = defaultdict(lambda: 0)
    print(buffer.this_frame_people)
    for name in buffer.this_frame_people:
        if sum(buffer.presences[name]) < len(buffer.presences[name]) / 2:
            classes[name] = AttentionClass.UNKNOWN
            scores[name] = -1
            continue
        mean_var = np.mean(buffer.lip_variances[name]) if len(buffer.lip_variances[name]) != 0 else 0
        mean_orient = np.mean(buffer.orientation_scores[name]) if len(buffer.orientation_scores[name]) != 0 else 0.5
        if any(buffer.yawns[name]):
            classes[name] = AttentionClass.DROWSY
            print("{} : DROWSY".format(name))
        elif ((mean_var > 100) or any(buffer.nods[name])) and mean_orient >= 0.6:
            classes[name] = AttentionClass.INTERACTIVE
            print("{} : INTERACTIVE".format(name))
        elif mean_orient >= 0.6:
            classes[name] = AttentionClass.ATTENTIVE
            print("{} : ATTENTIVE".format(name))
        else:
            classes[name] = AttentionClass.INATTENTIVE
            print("{} : INATTENTIVE".format(name))
        scores[name] = ((mean_var / 200) + 1 * mean_orient + 1 * (any(buffer.nods[name])) - 1 * any(
            buffer.yawns[name])) * 90
        # print("{} : {} : {}".format(name, classes[name], scores[name]))
        buffer.add_attention_score(name, scores[name])
        buffer.add_attention_class(name, classes[name])
        for name in buffer.all_people:
            if name not in buffer.this_frame_people:
                if sum(buffer.presences[name]) < len(buffer.presences[name]) / 2:
                    classes[name] = AttentionClass.UNKNOWN
                    scores[name] = -1
                    continue
                else:
                    classes[name] = buffer.attention_classes[name][-1]
                    scores[name] = buffer.attention_scores[name][-1]

    return classes, scores
