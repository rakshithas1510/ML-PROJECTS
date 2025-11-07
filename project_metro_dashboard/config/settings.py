import os


def getenv(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None else default


APP_PORT = int(getenv("APP_PORT", "8000"))
ES_HOST = getenv("ES_HOST", "http://elasticsearch:9200")
KIBANA_HOST = getenv("KIBANA_HOST", "http://kibana:5601")
PROM_SCRAPE_PATH = getenv("PROM_SCRAPE_PATH", "/metrics")
DATA_MODE = getenv("DATA_MODE", "sim")
SIM_TRAIN_COUNT = int(getenv("SIM_TRAIN_COUNT", "10"))
SIM_STATION_COUNT = int(getenv("SIM_STATION_COUNT", "20"))
TRAIN_CAPACITY = int(getenv("TRAIN_CAPACITY", "200"))
OPTIMIZE_INTERVAL_SECONDS = int(getenv("OPTIMIZE_INTERVAL_SECONDS", "60"))
ES_ENABLED = getenv("ES_ENABLED", "true").lower() == "true"
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")


ELASTIC_INDICES = {
    "trains": "metro-train-events",
    "stations": "metro-station-events",
    "routes": "metro-route-plans",
    "kpis": "metro-kpis",
}

