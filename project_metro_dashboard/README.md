# Metro Operations Dashboard

Real-time metro operations monitoring and analytics using Prometheus and ELK Stack.

## Features

- Simulated real-time train and station telemetry
- Flask API: `/trains`, `/stations`, `/routes`, `/stats`, `/metrics`
- KPIs aggregation and Prometheus metrics export
- Elasticsearch structured logs and Kibana-ready indices
- Prometheus alert rules for delays and overcrowding

## Quick Start

### Prerequisites

- Docker and Docker Compose

### Setup

Copy environment file (if your environment allows dotfiles):

```bash
cp .env.example .env
```

If you cannot create dotfiles, set env vars in `docker-compose.yml` (already defaults provided).

Build and run:

```bash
docker-compose up --build
```

### Usage

```bash
curl http://localhost:8000/trains
curl http://localhost:8000/stations
curl http://localhost:8000/stats
curl http://localhost:8000/metrics
```

### Services

- App: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`

### Kibana Index Patterns

- `metro-train-events*`
- `metro-station-events*`
- `metro-route-plans*`
- `metro-kpis*`

### Sample Visualizations

- Map of train locations (geo points)
- Delay trend line chart
- Passenger load per train (bar)
- Station occupancy (heat map)

### Prometheus Alerts

Alert rules in `docker/prometheus.rules.yml`:

- HighDelay: `metro_avg_delay_min > 5 for 10m`
- NoTrainData: `rate(metro_events_ingested_total[5m]) == 0`
- OvercrowdedStations: `metro_crowded_stations > 3 for 10m`

In Prometheus UI, check `/targets` and `/alerts` pages.

## Development

### Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Tests

```bash
pytest -q
```

## Troubleshooting

- Increase ES heap maps (Linux):
  ```bash
  sudo sysctl -w vm.max_map_count=262144
  ```
- Reset indices:
  ```bash
  curl -XDELETE 'http://localhost:9200/metro-*'
  ```

## Configuration Defaults

- APP_PORT=8000
- ES_HOST=http://elasticsearch:9200
- KIBANA_HOST=http://kibana:5601
- PROM_SCRAPE_PATH=/metrics
- DATA_MODE=sim
- SIM_TRAIN_COUNT=10
- SIM_STATION_COUNT=20
- TRAIN_CAPACITY=200
- OPTIMIZE_INTERVAL_SECONDS=60


