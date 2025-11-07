import time
from typing import Dict

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response


events_ingested = Counter("metro_events_ingested_total", "Total events ingested")
trips_completed = Counter("metro_trips_completed_total", "Total trips completed")

trains_active_g = Gauge("metro_trains_active", "Number of active trains")
avg_delay_g = Gauge("metro_avg_delay_min", "Average train delay in minutes")
total_passengers_g = Gauge("metro_total_passengers", "Total passenger count")
crowded_stations_g = Gauge("metro_crowded_stations", "Number of crowded stations")

response_hist = Histogram(
    "metro_response_time_seconds", "API response time", buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)


def observe_request(duration_seconds: float) -> None:
    response_hist.observe(duration_seconds)


def update_from_stats(stats: Dict) -> None:
    trains_active_g.set(stats.get("trains_active", 0))
    avg_delay_g.set(stats.get("avg_delay_min", 0.0))
    total_passengers_g.set(stats.get("passengers_total", 0))
    crowded_stations_g.set(stats.get("crowded_stations", 0))


def metrics_endpoint() -> Response:
    output = generate_latest()
    return Response(output, mimetype=CONTENT_TYPE_LATEST)

