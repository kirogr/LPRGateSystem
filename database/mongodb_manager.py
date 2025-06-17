from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import pymongo
from datetime import datetime
from database.models import VehicleEvent
from utils.logger import get_logger

class MongoDBManager:
    def __init__(self, config: dict):
        self.logger = get_logger(__name__)
        self.config = config['mongodb']
        self.processed_events = set()
        self._connect()

    def _connect(self):
        try:
            self.mongo_client = MongoClient(self.config['connection_string'])
            self.mongo_client.admin.command('ping')

            self.db = self.mongo_client[self.config['database_name']]
            self.vehicle_events_collection = self.db[self.config['vehicle_events_collection']]
            self.openalpr_results_collection = self.db[self.config['openalpr_results_collection']]

            self._setup_indexes()
            self.logger.info(f"MongoDB connection established: {self.config['database_name']}")

        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

    def _setup_indexes(self):
        try:
            self.vehicle_events_collection.create_index([
                ("track_id", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])
            self.vehicle_events_collection.create_index([
                ("license_plate", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])

            self.openalpr_results_collection.create_index([
                ("track_id", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])

            self.logger.info("MongoDB indexes created successfully")

        except Exception as e:
            self.logger.warning(f"Failed to create some indexes: {e}")

    def log_vehicle_event(self, event: VehicleEvent):
        event_key = f"{event.track_id}_{event.action}_{event.license_plate}"
        if event_key in self.processed_events:
            return

        try:
            document = {
                "timestamp": event.timestamp,
                "license_plate": event.license_plate,
                "action": event.action,
                "confidence": event.confidence,
                "track_id": event.track_id,
                "created_at": datetime.now()
            }

            result = self.vehicle_events_collection.insert_one(document)
            self.processed_events.add(event_key)
            self.logger.info(f"Logged event: {event.license_plate} - {event.action}, ObjectId: {result.inserted_id}")

        except Exception as e:
            self.logger.error(f"MongoDB error while logging event: {e}")

    def save_openalpr_results(self, timestamp: str, best_plate: str, best_confidence: float,
                              direction: str, snapshot_path: str, alpr_results: dict, track_id: int):
        try:
            document = {
                "timestamp": timestamp,
                "best_license_plate": best_plate,
                "best_confidence": best_confidence,
                "direction": direction,
                "snapshot_path": snapshot_path,
                "alpr_results": alpr_results,
                "track_id": track_id,
                "created_at": datetime.now()
            }

            result = self.openalpr_results_collection.insert_one(document)
            self.logger.info(f"Saved OpenALPR results for plate: {best_plate} with confidence {best_confidence}, ObjectId: {result.inserted_id}")

        except OperationFailure as e:
            self.logger.error(f"MongoDB operation failed while saving OpenALPR results: {e}")
        except Exception as e:
            self.logger.error(f"Failed to save OpenALPR results to MongoDB: {e}")
