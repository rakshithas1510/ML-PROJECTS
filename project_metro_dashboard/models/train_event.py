from datetime import datetime
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class TrainEvent(BaseModel):
    id: str
    train_id: str
    location: Tuple[float, float] = Field(description="(lat, lon)")
    speed_kmph: float
    ts: datetime
    delay_min: float = 0.0
    passenger_count: int = 0
    next_station: Optional[str] = None
    status: str = "running"
    capacity: int = 200

    @property
    def is_delayed(self) -> bool:
        return self.delay_min > 3.0

    @property
    def is_overcrowded(self) -> bool:
        return self.passenger_count > self.capacity

