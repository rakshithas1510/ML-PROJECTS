# Kibana Setup Guide

## Recommended Data Views

Create the following data views in Kibana for optimal visualization:

### 1. Metro Train Events
- **Name:** `metro-train-events`
- **Index Pattern:** `metro-train-events*`
- **Timestamp Field:** `ts`
- **Use for:** Train locations, speeds, delays, passenger counts

### 2. Metro Station Events
- **Name:** `metro-station-events`
- **Index Pattern:** `metro-station-events*`
- **Timestamp Field:** `ts`
- **Use for:** Platform occupancy, wait times, station alerts

### 3. Metro Route Plans
- **Name:** `metro-route-plans`
- **Index Pattern:** `metro-route-plans*`
- **Timestamp Field:** `ts_generated`
- **Use for:** Route optimization, efficiency scores, delay scores

### 4. Metro KPIs
- **Name:** `metro-kpis`
- **Index Pattern:** `metro-kpis*`
- **Timestamp Field:** `ts`
- **Use for:** Aggregated statistics, trends, dashboards

## Quick Setup Steps

1. In Kibana, go to **Stack Management** → **Data Views** (or **Index Patterns** in older versions)
2. Click **Create data view**
3. For each data view above:
   - Enter the **Name**
   - Enter the **Index pattern**
   - Select the **Timestamp field** from the dropdown
   - Click **Save data view to Kibana**

## Sample Visualizations

### 1. Train Location Map
- **Data View:** `metro-train-events`
- **Visualization:** Coordinate Map
- **Fields:** `location` (geo_point - you may need to create a scripted field or use lat/lon separately)

### 2. Delay Trend
- **Data View:** `metro-train-events` or `metro-kpis`
- **Visualization:** Line Chart
- **X-axis:** `ts` (time)
- **Y-axis:** `avg_delay_min` or `delay_min` (average)

### 3. Passenger Load
- **Data View:** `metro-train-events`
- **Visualization:** Bar Chart
- **X-axis:** `train_id`
- **Y-axis:** `passenger_count` (average)

### 4. Station Occupancy Heatmap
- **Data View:** `metro-station-events`
- **Visualization:** Heat Map
- **X-axis:** `station_id`
- **Y-axis:** `platform_occupancy` (average)

### 5. KPI Dashboard
- **Data View:** `metro-kpis`
- **Visualization:** Metric
- **Metrics:** 
  - `trains_active`
  - `avg_delay_min`
  - `passengers_total`
  - `on_time_percent`

## Notes

- All timestamp fields are in UTC
- Data is continuously being ingested (every 2 seconds for trains/stations)
- KPIs are updated every 15 seconds
- Use time range filters: Last 15 minutes, Last 1 hour, etc.

