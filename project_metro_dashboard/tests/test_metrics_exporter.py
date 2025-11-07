from services.metrics_exporter import (
    avg_delay_g,
    crowded_stations_g,
    total_passengers_g,
    trains_active_g,
    update_from_stats,
)


def test_metrics_update_from_stats():
    stats = {
        "trains_active": 5,
        "avg_delay_min": 2.5,
        "passengers_total": 800,
        "crowded_stations": 2,
    }
    update_from_stats(stats)
    assert trains_active_g._value.get() == 5  # type: ignore[attr-defined]
    assert avg_delay_g._value.get() == 2.5  # type: ignore[attr-defined]
    assert total_passengers_g._value.get() == 800  # type: ignore[attr-defined]
    assert crowded_stations_g._value.get() == 2  # type: ignore[attr-defined]

