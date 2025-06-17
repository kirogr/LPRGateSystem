import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, List
from core.detection.zone_detector import DetectionZone
from utils.logger import get_logger


class VehicleTracker:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.tracks = {}
        self.position_history = defaultdict(lambda: deque(maxlen=30))
        self.gate_crossings = {}
        self.parked_vehicles = set()
        self.plate_roi_temp = defaultdict(lambda: deque(maxlen=30))
        self.ocr_done = set()
        self.ocr_scheduled = {}

        self.max_parking_movement = 20
        self.min_parking_time = 2.5
        self.parking_check_interval = 10

    def update_track(self, track_id: int, bbox: Tuple[int, int, int, int],
                     license_plate: str = None, confidence: float = 0.0):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_time = time.time()

        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'license_plates': [],
                'positions': deque(maxlen=15),
                'state': 'UNKNOWN',
                'zone_history': [],
                'best_license_plate': None,
                'best_confidence': 0.0,
                'stationary_start_time': None,
                'last_position': None,
                'movement_sum': 0.0,
                'position_count': 0
            }

        track_data = self.tracks[track_id]
        track_data['last_seen'] = current_time

        if track_data['last_position'] is not None:
            last_x, last_y = track_data['last_position']
            distance_moved = abs(center_x - last_x) + abs(center_y - last_y)

            track_data['movement_sum'] += distance_moved
            track_data['position_count'] += 1

            if distance_moved > self.max_parking_movement:
                track_data['stationary_start_time'] = None
                if track_id in self.parked_vehicles:
                    self.parked_vehicles.discard(track_id)
            elif track_data['stationary_start_time'] is None and distance_moved < 5:
                track_data['stationary_start_time'] = current_time

        track_data['last_position'] = (center_x, center_y)
        track_data['positions'].append((center_x, center_y, current_time))

        if license_plate and confidence > track_data['best_confidence']:
            track_data['best_license_plate'] = license_plate
            track_data['best_confidence'] = confidence
            track_data['license_plates'].append((license_plate, confidence, current_time))

    def get_movement_direction(self, track_id: int) -> Optional[str]:
        if track_id not in self.tracks:
            return None

        positions = self.tracks[track_id]['positions']
        if len(positions) < 5:
            return None

        first_y = positions[0][1]
        last_y = positions[-1][1]
        y_trend = last_y - first_y

        if y_trend > 25:
            return 'ENTER'
        elif y_trend < -25:
            return 'LEAVE'

        return None

    def is_parked(self, track_id: int, parking_zones: List[DetectionZone]) -> bool:
        if track_id not in self.tracks:
            return False

        track_data = self.tracks[track_id]
        current_time = time.time()

        if track_data['stationary_start_time'] is None:
            return False

        stationary_duration = current_time - track_data['stationary_start_time']
        if stationary_duration < self.min_parking_time:
            return False

        if track_data['position_count'] > 5:
            avg_movement = track_data['movement_sum'] / track_data['position_count']
            if avg_movement > self.max_parking_movement / 2:
                return False

        if track_data['last_position']:
            x, y = track_data['last_position']
            for zone in parking_zones:
                if zone.contains_point(x, y):
                    return True

        return False

    def get_track_movement_status(self, track_id: int) -> str:
        if track_id not in self.tracks:
            return "UNKNOWN"

        track_data = self.tracks[track_id]

        if track_data['stationary_start_time']:
            duration = time.time() - track_data['stationary_start_time']
            if duration > self.min_parking_time:
                return "PARKED"
            else:
                return "STOPPING"

        return "MOVING"

    def cleanup_old_data(self, track_id: int):
        if track_id in self.tracks:
            track_data = self.tracks[track_id]
            if track_data['position_count'] > 50:
                track_data['movement_sum'] = track_data['movement_sum'] * 0.5
                track_data['position_count'] = 25