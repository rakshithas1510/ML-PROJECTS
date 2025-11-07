import threading
from collections import deque
from datetime import datetime, timezone
from statistics import mean
from typing import Deque, Dict, List, Tuple, Union

from models.station_event import StationEvent
from models.train_event import TrainEvent


Event = Union[TrainEvent, StationEvent]


class Aggregator:
    def __init__(self, window_size: int = 300) -> None:
        self.lock = threading.RLock()
        self.trains: Dict[str, TrainEvent] = {}
        self.stations: Dict[str, StationEvent] = {}
        self.window: Deque[Tuple[datetime, Event]] = deque(maxlen=window_size)

    def update(self, event: Event) -> None:
        with self.lock:
            self.window.append((event.ts, event))
            if isinstance(event, TrainEvent):
                self.trains[event.train_id] = event
            elif isinstance(event, StationEvent):
                self.stations[event.station_id] = event

    def compute_stats(self) -> Dict:
        with self.lock:
            trains_list: List[TrainEvent] = list(self.trains.values())
            stations_list: List[StationEvent] = list(self.stations.values())
            trains_active = len(trains_list)
            avg_delay = mean([t.delay_min for t in trains_list]) if trains_list else 0.0
            avg_speed = mean([t.speed_kmph for t in trains_list]) if trains_list else 0.0
            total_passengers = sum([t.passenger_count for t in trains_list])
            crowded_stations = sum([1 for s in stations_list if s.platform_occupancy > 350])
            missed_stops = 0  # Placeholder for advanced logic
            on_time_percent = (
                100.0
                * (
                    sum(1 for t in trains_list if not t.is_delayed) / trains_active
                    if trains_active
                    else 0.0
                )
            )
            return {
                "ts": datetime.now(timezone.utc).isoformat(),
                "trains_active": trains_active,
                "avg_delay_min": round(avg_delay, 2),
                "avg_speed_kmph": round(avg_speed, 2),
                "passengers_total": total_passengers,
                "crowded_stations": crowded_stations,
                "missed_stops": missed_stops,
                "on_time_percent": round(on_time_percent, 2),
            }

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                "trains": [t.model_dump() for t in self.trains.values()],
                "stations": [s.model_dump() for s in self.stations.values()],
            }

