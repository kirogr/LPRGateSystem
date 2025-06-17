from dataclasses import dataclass
from datetime import datetime

@dataclass
class VehicleEvent:
    timestamp: datetime
    license_plate: str
    action: str  # 'ENTER' / 'LEAVE'
    confidence: float
    track_id: int