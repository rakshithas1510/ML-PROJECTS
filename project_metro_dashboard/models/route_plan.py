from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel


class RoutePlan(BaseModel):
    id: str
    ts_generated: datetime
    train_id: str
    planned_stops: List[str]
    actual_stops: List[str] = []
    total_distance_km: float = 0.0
    delay_score: float = 0.0
    efficiency_score: float = 1.0

    def summarize(self) -> Tuple[int, int]:
        return len(self.planned_stops), len(self.actual_stops)

