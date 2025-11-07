from datetime import datetime
from typing import List

from pydantic import BaseModel


class StationEvent(BaseModel):
    id: str
    station_id: str
    ts: datetime
    platform_occupancy: int
    avg_wait_min: float
    alerts: List[str] = []

