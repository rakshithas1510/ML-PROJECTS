import asyncio
import math
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from geopy.distance import geodesic

from config import settings
from models.station_event import StationEvent
from models.train_event import TrainEvent


CityBounds = Tuple[Tuple[float, float], Tuple[float, float]]


def _random_point(bounds: CityBounds) -> Tuple[float, float]:
    (min_lat, min_lon), (max_lat, max_lon) = bounds
    return (
        random.uniform(min_lat, max_lat),
        random.uniform(min_lon, max_lon),
    )


class Simulator:
    def __init__(
        self,
        train_count: int,
        station_count: int,
        train_capacity: int,
        city_bounds: CityBounds = ((28.40, 77.00), (28.90, 77.40)),
        tick_seconds: float = 2.0,
    ) -> None:
        self.train_count = train_count
        self.station_count = station_count
        self.train_capacity = train_capacity
        self.city_bounds = city_bounds
        self.tick_seconds = tick_seconds
        self.station_ids: List[str] = [f"STN-{i:03d}" for i in range(station_count)]
        self.train_ids: List[str] = [f"TRN-{i:03d}" for i in range(train_count)]
        self.train_state: Dict[str, Dict] = {}

    def _init_trains(self) -> None:
        for t in self.train_ids:
            start = _random_point(self.city_bounds)
            planned_stops = random.sample(self.station_ids, k=min(10, len(self.station_ids)))
            self.train_state[t] = {
                "location": start,
                "speed_kmph": random.uniform(20.0, 60.0),
                "delay_min": random.uniform(-1.0, 8.0),
                "passenger_count": random.randint(20, self.train_capacity),
                "next_station": random.choice(planned_stops),
                "status": "running",
                "route": planned_stops,
                "capacity": self.train_capacity,
            }

    async def run(self, queue: asyncio.Queue) -> None:
        self._init_trains()
        while True:
            ts = datetime.now(timezone.utc)
            # Update trains
            for train_id in self.train_ids:
                st = self.train_state[train_id]
                lat, lon = st["location"]
                # Move by small random delta
                bearing = random.uniform(0, 360)
                step_km = st["speed_kmph"] * self.tick_seconds / 3600.0
                dlat = (step_km / 111.0) * math.cos(math.radians(bearing))
                dlon = (step_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(
                    math.radians(bearing)
                )
                new_loc = (lat + dlat, lon + dlon)
                # Wrap within bounds lightly
                (min_lat, min_lon), (max_lat, max_lon) = self.city_bounds
                new_loc = (
                    max(min_lat, min(max_lat, new_loc[0])),
                    max(min_lon, min(max_lon, new_loc[1])),
                )
                st["location"] = new_loc
                st["speed_kmph"] = max(0.0, min(80.0, st["speed_kmph"] + random.uniform(-2, 2)))
                st["delay_min"] += random.uniform(-0.2, 0.5)
                # Passenger churn
                delta_p = random.randint(-10, 12)
                st["passenger_count"] = max(0, st["passenger_count"] + delta_p)
                # Occasionally advance to next stop
                if random.random() < 0.15 and st["route"]:
                    st["next_station"] = random.choice(st["route"])  # simplification

                ev = TrainEvent(
                    id=str(uuid.uuid4()),
                    train_id=train_id,
                    location=new_loc,
                    speed_kmph=st["speed_kmph"],
                    ts=ts,
                    delay_min=max(-2.0, st["delay_min"]),
                    passenger_count=st["passenger_count"],
                    next_station=st.get("next_station"),
                    status=st.get("status", "running"),
                    capacity=st.get("capacity", self.train_capacity),
                )
                await queue.put(ev)

            # Update a subset of stations
            for stn in random.sample(self.station_ids, k=min(5, len(self.station_ids))):
                occ = random.randint(0, 500)
                avg_wait = max(0.0, random.gauss(3.5, 1.0))
                alerts: List[str] = []
                if occ > 350:
                    alerts.append("High occupancy")
                if avg_wait > 6:
                    alerts.append("Long wait")
                sev = StationEvent(
                    id=str(uuid.uuid4()),
                    station_id=stn,
                    ts=ts,
                    platform_occupancy=occ,
                    avg_wait_min=avg_wait,
                    alerts=alerts,
                )
                await queue.put(sev)

            await asyncio.sleep(self.tick_seconds)


async def simulate(queue: asyncio.Queue) -> None:
    sim = Simulator(
        train_count=settings.SIM_TRAIN_COUNT,
        station_count=settings.SIM_STATION_COUNT,
        train_capacity=settings.TRAIN_CAPACITY,
    )
    await sim.run(queue)

