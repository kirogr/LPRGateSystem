import numpy as np
from typing import List, Tuple
from utils.logger import get_logger

class SimpleTrack:
    def __init__(self, track_id: int, centroid: Tuple[int, int]):
        self.track_id = track_id
        self.centroid = centroid

    def is_confirmed(self):
        return True

    def to_ltrb(self):
        cx, cy = self.centroid
        w, h = 80, 120
        return [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]

class SimpleTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 100):
        self.logger = get_logger(__name__)
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int]):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> List[SimpleTrack]:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])

        tracks = []
        for object_id, centroid in self.objects.items():
            tracks.append(SimpleTrack(object_id, centroid))

        return tracks