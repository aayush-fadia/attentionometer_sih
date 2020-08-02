from collections import defaultdict, deque
from threading import RLock

import Levenshtein


class Buffer:
    def __init__(self, MAX_FRAMES=20):
        self.lip_distances = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.orientation_vectors = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.attention_scores = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.lip_variances = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.yawns = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.nods = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.presences = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        self.num_people = -1
        self.all_people = set()
        self.this_frame_people = set()
        self.lock = RLock()

    def add_lip_dist(self, name, dist):
        self.lip_distances[name].append(dist)

    def add_orientation_vector(self, name, vect):
        self.orientation_vectors[name].append(vect)

    def add_attention_score(self, name, score):
        self.attention_scores[name].append(score)

    def add_variance(self, name, variance):
        self.lip_variances[name].append(variance)

    def add_yawn(self, name, yawn):
        self.yawns[name].append(yawn)

    def add_nod(self, name, nod):
        self.nods[name].append(nod)

    def set_num_people(self, num_people):
        self.num_people = num_people

    def announce(self, name):
        self.this_frame_people.add(name)
        self.all_people.add(name)

    def reset_people(self):
        self.this_frame_people = set()

    def set_pressences(self):
        for name in self.all_people:
            if name not in self.this_frame_people:
                self.presences[name].append(False)
            else:
                self.presences[name].append(True)

    def match_name(self, name):
        parts = name.split()
        for n in self.all_people:
            n_split = n.split()
            if Levenshtein.distance(n, name) < 5 or Levenshtein.distance(parts[0], n_split[0]) < 2:
                return n
        return name
