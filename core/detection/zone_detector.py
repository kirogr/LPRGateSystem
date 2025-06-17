import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class DetectionZone:
    name: str
    polygon: List[Tuple[int, int]]

    def contains_point(self, x: int, y: int) -> bool:
        return cv2.pointPolygonTest(np.array(self.polygon), (x, y), False) >= 0

    def create_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.polygon, np.int32)], 255)
        return mask


class ZoneManager:
    def __init__(self, config: dict):
        self.parking_zones = self._create_zones(config['parking_zones'])
        self.detection_zones = self._create_zones(config['detection_zones'])
        self.gate_zone = self._find_zone('gate_1')
        self.ocr_zone = self._find_zone('detection_2')

    def _create_zones(self, zone_configs: List[dict]) -> List[DetectionZone]:
        return [DetectionZone(zc['name'], zc['points']) for zc in zone_configs]

    def _find_zone(self, name: str) -> DetectionZone:
        for zone in self.detection_zones:
            if zone.name == name:
                return zone
        return None

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        for zone in self.parking_zones:
            pts = np.array(zone.polygon, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for zone in self.detection_zones:
            pts = np.array(zone.polygon, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame