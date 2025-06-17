import cv2
import numpy as np
import easyocr
import random
from typing import List, Tuple, Optional
from utils.logger import get_logger


class LicensePlateRecognizer:
    def __init__(self, ocr_confidence: float = 0.6):
        self.logger = get_logger(__name__)
        self.ocr_reader = easyocr.Reader(['en'])
        self.ocr_confidence = ocr_confidence

    def is_valid_license_plate(self, text: str) -> bool:
        if not text:
            return False

        clean_text = ''.join(c for c in text.upper() if c.isalnum())

        if not (4 <= len(clean_text) <= 9):
            return False

        has_digit = any(c.isdigit() for c in clean_text)
        has_letter = any(c.isalpha() for c in clean_text)

        if not has_digit and len(clean_text) > 6:
            return False

        return True

    def extract_license_plate_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return frame[y1:y2, x1:x2]

    def recognize_license_plate(self, roi_list: List[np.ndarray], direction: str,
                                track_id: int = None) -> Tuple[Optional[str], float]:
        if not roi_list:
            return None, 0.0

        try:
            best_plate = None
            best_confidence = 0.0

            sample_size = min(3, len(roi_list))
            sample_rois = random.sample(roi_list, sample_size)

            for selected_roi in sample_rois:
                cv2.imshow(f"License Plate ROI - Track {track_id} - Direction {direction}", selected_roi)
                if selected_roi is None or selected_roi.size == 0:
                    continue

                plate, confidence = self._process_single_roi(selected_roi)

                if plate and self.is_valid_license_plate(plate) and confidence > best_confidence:
                    best_plate = plate
                    best_confidence = confidence



            if best_plate and best_confidence > self.ocr_confidence:
                self.logger.info(f"Final recognized plate: {best_plate} with confidence {best_confidence}")
                return best_plate.upper(), best_confidence
            else:
                self.logger.warning(f"No valid license plate detected for track {track_id} in direction {direction}")

        except Exception as e:
            self.logger.error(f"OCR error: {e}")

        return None, 0.0

    def _process_single_roi(self, roi: np.ndarray) -> Tuple[Optional[str], float]:
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            results = self.ocr_reader.readtext(enhanced)
            best_text = ""
            best_confidence = 0.0

            for (bbox, text, confidence) in results:
                clean_text = ''.join(c for c in text if c.isalnum())
                if self.is_valid_license_plate(clean_text) and confidence > best_confidence:
                    best_text = clean_text
                    best_confidence = confidence

            return best_text.upper() if best_text else None, best_confidence

        except Exception as e:
            self.logger.error(f"Single ROI processing error: {e}")
            return None, 0.0