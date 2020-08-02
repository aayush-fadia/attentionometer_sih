from collections import defaultdict
from enum import Enum


class AttentionClass(Enum):
    UNKNOWN = 0
    DROWSY = 1
    INATTENTIVE = 2
    ATTENTIVE = 3
    INTERACTIVE = 4


def classify(buffer):
    classes = defaultdict(lambda: AttentionClass.UNKNOWN)
    for name in buffer.this_frame_people:
        if any(buffer.yawns[name]):
            classes[name] = AttentionClass.DROWSY
