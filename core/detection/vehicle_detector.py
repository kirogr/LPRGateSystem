import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
from utils.logger import get_logger

class VehicleDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.5):
        self.logger = get_logger(__name__)
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def detect(self, frame: np.ndarray, mask: np.ndarray = None) -> List[Tuple[int, int, int, int, float]]:
        vehicles = []

        detection_frame = frame
        if mask is not None:
            detection_frame = cv2.bitwise_and(frame, frame, mask=mask)

        results = self.model(detection_frame, conf=self.confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        vehicles.append((int(x1), int(y1), int(x2), int(y2), confidence))

        return vehicles