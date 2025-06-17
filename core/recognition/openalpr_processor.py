import os
import json
import subprocess
import threading
from queue import Queue
from datetime import datetime
from typing import Tuple, Optional
from utils.logger import get_logger


class OpenALPRProcessor:
    def __init__(self, openalpr_path: str, storage_folder: str = "vehicle_snapshots"):
        self.logger = get_logger(__name__)
        self.openalpr_path = openalpr_path
        self.storage_folder = storage_folder
        self.openalpr_queue = Queue()
        self.openalpr_processed = set()

        self._create_storage_folder()
        self._start_worker_thread()

    def _create_storage_folder(self):
        if not os.path.exists(self.storage_folder):
            os.makedirs(self.storage_folder)
            self.logger.info(f"Created storage folder: {self.storage_folder}")

    def _start_worker_thread(self):
        self.openalpr_thread = threading.Thread(target=self._openalpr_worker, daemon=True)
        self.openalpr_thread.start()

    def _openalpr_worker(self):
        while True:
            try:
                task = self.openalpr_queue.get()
                if task is None:
                    self.logger.info("Shutting down OpenALPR worker thread.")
                    break

                image_path, timestamp, direction, track_id, original_plate, original_confidence = task

                full_image_path = os.path.join(os.getcwd(), image_path)

                alpr_results = self.run_openalpr(full_image_path)
                if alpr_results:
                    alpr_best_plate, alpr_best_confidence = self.extract_best_plate_from_alpr(alpr_results)

                    self.save_openalpr_results(
                        timestamp=timestamp,
                        best_plate=alpr_best_plate or original_plate,
                        best_confidence=alpr_best_confidence if alpr_best_plate else original_confidence,
                        direction=direction,
                        snapshot_path=full_image_path,
                        alpr_results=alpr_results,
                        track_id=track_id
                    )

                    if track_id in self.tracker.tracks:
                        self.tracker.tracks[track_id]['best_license_plate'] = alpr_best_plate or original_plate
                        self.tracker.tracks[track_id]['best_confidence'] = alpr_best_confidence or original_confidence

                    self.logger.info(f"OpenALPR processing completed for track {track_id}")

                else:
                    self.logger.warning(f"OpenALPR returned no results for track {track_id}")

                self.openalpr_queue.task_done()

            except Exception as e:
                self.logger.error(f"OpenALPR worker error: {e}", exc_info=True)
                self.openalpr_queue.task_done()

            while True:
                try:
                    task = self.openalpr_queue.get()
                    if task is None:
                        break

                    image_path, timestamp, direction, track_id, original_plate, original_confidence = task
                    full_image_path = os.path.join(os.getcwd(), image_path)

                    alpr_results = self.run_openalpr(full_image_path)

                    if alpr_results:
                        alpr_best_plate, alpr_best_confidence = self.extract_best_plate_from_alpr(alpr_results)

                        self.save_openalpr_results(
                            timestamp=timestamp,
                            best_plate=alpr_best_plate or original_plate,
                            best_confidence=alpr_best_confidence if alpr_best_plate else original_confidence,
                            direction=direction,
                            snapshot_path=full_image_path,
                            alpr_results=alpr_results,
                            track_id=track_id
                        )

                        self.logger.info(f"OpenALPR background processing completed for track {track_id}")
                    else:
                        self.logger.warning(f"OpenALPR failed for track {track_id}")

                    self.openalpr_queue.task_done()

                except Exception as e:
                    self.logger.error(f"OpenALPR worker error: {e}")
                    self.openalpr_queue.task_done()

    def run_openalpr(self, image_path: str) -> dict:
        try:
            cmd = [self.openalpr_path, "-c", "eu", "-j", image_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                alpr_data = json.loads(result.stdout)
                self.logger.info(f"OpenALPR executed successfully for {image_path}")
                return alpr_data
            else:
                self.logger.error(f"OpenALPR failed with return code {result.returncode}: {result.stderr}")
                return {}

        except Exception as e:
            self.logger.error(f"OpenALPR execution error: {e}")
            return {}

    def extract_best_plate_from_alpr(self, alpr_results: dict) -> Tuple[Optional[str], float]:
        if not alpr_results or 'results' not in alpr_results:
            return None, 0.0

        best_plate = None
        best_confidence = 0.0

        for result in alpr_results['results']:
            if 'candidates' in result:
                for candidate in result['candidates']:
                    plate = candidate.get('plate', '').strip()
                    confidence = candidate.get('confidence', 0.0)

                    if confidence < 83.0:
                        continue

                    if any(c.isalpha() and c.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in plate):
                        continue

                    if confidence > best_confidence:
                        best_plate = plate
                        best_confidence = confidence

        return best_plate, best_confidence

    def queue_openalpr_processing(self, image_path: str, timestamp: str, direction: str,
                                  track_id: int, original_plate: str, original_confidence: float):
        try:
            task = (image_path, timestamp, direction, track_id, original_plate, original_confidence)
            self.openalpr_queue.put(task)
            self.logger.info(f"Queued OpenALPR processing for track {track_id} - {direction}")
        except Exception as e:
            self.logger.error(f"Failed to queue OpenALPR processing: {e}")