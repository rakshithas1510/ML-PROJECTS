import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any

from flask import Flask, jsonify

from config import settings
from models.train_event import TrainEvent
from services.aggregator import Aggregator
from services.elastic_logger import make_es_logger
from services.metrics_exporter import events_ingested, metrics_endpoint, observe_request, update_from_stats
from services.schedule_optimizer import ScheduleOptimizer
from services.scheduler import AppScheduler
from services.simulator import simulate


logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("app")

app = Flask(__name__)


aggregator = Aggregator()
es_logger = make_es_logger()
optimizer = ScheduleOptimizer()


async def consumer(queue: asyncio.Queue) -> None:
    while True:
        ev = await queue.get()
        start = time.time()
        try:
            # Update aggregator and metrics
            aggregator.update(ev)  # type: ignore[arg-type]
            events_ingested.inc()
            # Log to Elasticsearch with dynamic index
            index_name = None
            if hasattr(ev, "train_id"):
                index_name = settings.ELASTIC_INDICES["trains"]
            elif hasattr(ev, "station_id"):
                index_name = settings.ELASTIC_INDICES["stations"]
            if index_name:
                es_logger.index(index_name, ev.model_dump())  # type: ignore[attr-defined]
        finally:
            observe_request(time.time() - start)
            queue.task_done()


def start_background_simulation() -> None:
    loop = asyncio.new_event_loop()

    async def runner() -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        cons = asyncio.create_task(consumer(queue))
        prod = asyncio.create_task(simulate(queue))
        await asyncio.gather(cons, prod)

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner())

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()


def run_optimizer_once() -> None:
    # Build inputs from aggregator state
    snapshot = aggregator.snapshot()
    delays = {t["train_id"]: t["delay_min"] for t in snapshot["trains"]}
    routes = {t["train_id"]: [t.get("next_station")] for t in snapshot["trains"]}
    plans = optimizer.optimize(delays, routes)
    for p in plans:
        es_logger.index(settings.ELASTIC_INDICES["routes"], p.model_dump())


# Initialize background services
_bootstrap_done = False
_bootstrap_lock = threading.Lock()


def _bootstrap() -> None:
    global _bootstrap_done
    with _bootstrap_lock:
        if _bootstrap_done:
            return
        start_background_simulation()
        scheduler = AppScheduler(aggregator.compute_stats, run_optimizer_once)
        scheduler.start()
        _bootstrap_done = True


@app.route("/", methods=["GET"])
def root():
    _bootstrap()
    return jsonify({
        "service": "Metro Operations Dashboard",
        "version": "1.0.0",
        "endpoints": {
            "/trains": "GET - Current status of all trains",
            "/stations": "GET - Platform occupancy and alerts for all stations",
            "/routes": "GET - Route plans information",
            "/stats": "GET - KPIs and summary statistics",
            "/metrics": "GET - Prometheus metrics endpoint"
        },
        "status": "operational",
        "docs": "See README.md for usage examples"
    })


@app.route("/trains", methods=["GET"])
def get_trains():
    _bootstrap()
    stats = aggregator.snapshot()
    return jsonify(stats.get("trains", []))


@app.route("/stations", methods=["GET"])
def get_stations():
    _bootstrap()
    stats = aggregator.snapshot()
    return jsonify(stats.get("stations", []))


@app.route("/routes", methods=["GET"])
def get_routes():
    _bootstrap()
    # This demo exposes last optimized plans via ES is optional; return empty for now
    return jsonify({"message": "Route plans are logged to Elasticsearch index metro-route-plans"})


@app.route("/stats", methods=["GET"])
def get_stats():
    _bootstrap()
    stats = aggregator.compute_stats()
    update_from_stats(stats)
    es_logger.index(settings.ELASTIC_INDICES["kpis"], stats)
    return jsonify(stats)


@app.route(settings.PROM_SCRAPE_PATH, methods=["GET"])  # /metrics
def metrics():
    _bootstrap()
    return metrics_endpoint()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.APP_PORT)

