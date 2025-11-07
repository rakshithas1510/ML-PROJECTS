import asyncio

import pytest

from services.simulator import Simulator


@pytest.mark.asyncio
async def test_simulator_generates_valid_events():
    sim = Simulator(train_count=2, station_count=3, train_capacity=150, tick_seconds=0.01)
    q: asyncio.Queue = asyncio.Queue()

    async def run_once():
        # Run a single tick
        sim._init_trains()
        ts = asyncio.create_task(sim.run(q))
        # collect few events then cancel
        events = []
        for _ in range(10):
            ev = await q.get()
            events.append(ev)
            q.task_done()
        ts.cancel()
        return events

    events = await run_once()
    assert len(events) > 0
    any_train = None
    for ev in events:
        if hasattr(ev, "train_id"):
            any_train = ev
            break
    assert any_train is not None
    lat, lon = any_train.location  # type: ignore[union-attr]
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0
    assert any_train.speed_kmph >= 0  # type: ignore[union-attr]

