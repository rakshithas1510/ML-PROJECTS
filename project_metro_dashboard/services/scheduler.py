import logging
from datetime import datetime, timezone
from typing import Callable, Dict

from apscheduler.schedulers.background import BackgroundScheduler

from config import settings


logger = logging.getLogger("scheduler")


class AppScheduler:
    def __init__(self, aggregator_compute: Callable[[], Dict], optimizer_run: Callable[[], None]) -> None:
        self.scheduler = BackgroundScheduler()
        self.aggregator_compute = aggregator_compute
        self.optimizer_run = optimizer_run

    def start(self) -> None:
        # KPIs snapshot every 15s
        self.scheduler.add_job(self._snapshot_kpis, "interval", seconds=15, id="kpi_snapshot")
        # Optimizer
        self.scheduler.add_job(
            self._run_optimizer, "interval", seconds=settings.OPTIMIZE_INTERVAL_SECONDS, id="optimizer"
        )
        # Heartbeat
        self.scheduler.add_job(self._heartbeat, "interval", seconds=30, id="heartbeat")
        self.scheduler.start()

    def _snapshot_kpis(self) -> None:
        stats = self.aggregator_compute()
        logger.info("KPI snapshot: %s", stats)

    def _run_optimizer(self) -> None:
        logger.info("Running optimizer")
        self.optimizer_run()

    def _heartbeat(self) -> None:
        logger.info("App heartbeat %s", datetime.now(timezone.utc).isoformat())

