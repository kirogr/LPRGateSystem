import json
import os
from typing import Dict


class ConfigLoader:
    @staticmethod
    def load_config(config_file: str = "config/lpr_config.json") -> Dict:
        default_config = {
            "camera_source": 0,
            "detection_confidence": 0.5,
            "ocr_confidence": 0.6,
            "gate_position": {"x": 640, "y": 400},
            "parking_zones": [
                {"name": "parking_left", "points": [[50, 300], [300, 300], [300, 500], [50, 500]]},
                {"name": "parking_right", "points": [[900, 300], [1200, 300], [1200, 500], [900, 500]]}
            ],
            "detection_zones": [
                {"name": "gate_1", "points": [[400, 200], [800, 200], [800, 600], [400, 600]]},
                {"name": "detection_2", "points": [[500, 300], [700, 300], [700, 400], [500, 400]]}
            ],
            "min_stationary_time": 3.0,
            "max_wait_time": 180.0,
            "mongodb": {
                "connection_string": "mongodb://localhost:27017/",
                "database_name": "lpr_system",
                "vehicle_events_collection": "vehicle_events",
                "openalpr_results_collection": "openalpr_results"
            }
        }

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config