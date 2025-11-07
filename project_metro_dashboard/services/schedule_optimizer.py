from datetime import datetime, timezone
from typing import Dict, List

from models.route_plan import RoutePlan


class ScheduleOptimizer:
    def __init__(self) -> None:
        pass

    def optimize(self, train_delays: Dict[str, float], train_routes: Dict[str, List[str]]) -> List[RoutePlan]:
        plans: List[RoutePlan] = []
        for train_id, delay in train_delays.items():
            route = train_routes.get(train_id, [])
            # Simple heuristic: if delayed, reduce stops by skipping every other stop
            if delay > 5 and len(route) > 4:
                proposed = route[::2]
            else:
                proposed = route
            score = max(0.0, 1.0 - (delay / 30.0))
            plans.append(
                RoutePlan(
                    id=f"plan-{train_id}-{int(datetime.now(tz=timezone.utc).timestamp())}",
                    ts_generated=datetime.now(tz=timezone.utc),
                    train_id=train_id,
                    planned_stops=proposed,
                    total_distance_km=float(len(proposed)),
                    delay_score=float(delay),
                    efficiency_score=score,
                )
            )
        return plans

