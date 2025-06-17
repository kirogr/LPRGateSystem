import cv2
from datetime import datetime
from config.config_loader import ConfigLoader
from core.detection.vehicle_detector import VehicleDetector
from core.detection.zone_detector import ZoneManager
from core.tracking.simple_tracker import SimpleTracker
from core.tracking.vehicle_tracker import VehicleTracker
from core.recognition.license_plate_recognizer import LicensePlateRecognizer
from core.recognition.openalpr_processor import OpenALPRProcessor
from database.mongodb_manager import MongoDBManager
from database.models import VehicleEvent
from utils.logger import get_logger

class LPRGateSystem:
    def __init__(self):
        self.config = ConfigLoader.load_config()
        self.logger = get_logger(__name__)

        self.zone_manager = ZoneManager(self.config)
        self.detector = VehicleDetector(confidence=self.config['detection_confidence'])
        self.tracker = VehicleTracker()
        self.simple_tracker = SimpleTracker()
        self.plate_recognizer = LicensePlateRecognizer(ocr_confidence=self.config['ocr_confidence'])
        self.openalpr = OpenALPRProcessor(openalpr_path=r"alpr_binary/openalpr.exe")
        self.db = MongoDBManager(self.config)

        self.frame_count = 0
        self.out = cv2.VideoWriter('playback.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (1920, 1080))

    def run(self):
        cap = cv2.VideoCapture(self.config['camera_source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            self.logger.error("Failed to open video source")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Frame not captured")
                    break

                frame = cv2.resize(frame, (1920, 1080))
                processed = self.process_frame(frame)
                cv2.imshow("LPR Gate System", processed)
                self.out.write(processed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            cap.release()
            self.out.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame):
        self.frame_count += 1
        gate_mask = self.zone_manager.gate_zone.create_mask(frame.shape)
        detections = self.detector.detect(frame, mask=gate_mask)

        rects = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2, _ in detections]
        tracks = self.simple_tracker.update(rects)

        now = datetime.now()
        for track in tracks:
            tid = track.track_id
            bbox = tuple(map(int, track.to_ltrb()))
            self.tracker.update_track(tid, bbox)
            td = self.tracker.tracks[tid]

            if self.frame_count % 10 == 0 and self.zone_manager.ocr_zone.contains_point(*td['last_position']):
                roi = self.plate_recognizer.extract_license_plate_roi(frame, bbox)
                if roi is not None and roi.size > 0:
                    self.tracker.plate_roi_temp[tid].append(roi)

            direction = self.tracker.get_movement_direction(tid)
            if (tid not in self.tracker.ocr_done and direction and
                len(self.tracker.plate_roi_temp[tid]) >= 3):

                plate, conf = self.plate_recognizer.recognize_license_plate(
                    list(self.tracker.plate_roi_temp[tid]), direction, tid)

                if plate:
                    td['best_license_plate'] = plate
                    td['best_confidence'] = conf
                    self.tracker.ocr_done.add(tid)
                    self.db.log_vehicle_event(VehicleEvent(
                        timestamp=now,
                        license_plate=plate,
                        action=direction,
                        confidence=conf,
                        track_id=tid
                    ))
        return frame
