# Kibana Visualization Guide - Beautiful Graphs

## 📊 Best Field Combinations for Visualizations

### 🚂 Metro Train Events (`metro-train-events*`)

#### 1. **Delay Trend Over Time** (Line Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `delay_min` (Average)
- **Split by:** `train_id` (optional, for multiple lines)
- **Color:** Gradient by `train_id`
- **Time Range:** Last 15 minutes or Last 1 hour

#### 2. **Train Speed Distribution** (Area Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `speed_kmph` (Average)
- **Split by:** `train_id`
- **Color:** Different color per train

#### 3. **Passenger Load by Train** (Bar Chart)
- **X-axis:** `train_id` (Terms)
- **Y-axis:** `passenger_count` (Average)
- **Color:** Gradient by `passenger_count`
- **Sort:** Descending by passenger count

#### 4. **Capacity Utilization** (Gauge/Donut)
- **Metric:** `passenger_count` (Average) / `capacity` × 100
- **Range:** 0-100%
- **Color bands:** Green (0-70%), Yellow (70-90%), Red (90-100%)

#### 5. **Delayed vs On-Time Trains** (Pie Chart)
- **Field:** Create scripted field `is_delayed` (delay_min > 3)
- **Slice by:** `is_delayed` (Terms)
- **Size:** Count of documents

#### 6. **Train Location Map** (Coordinate Map)
- **Field:** `location` (Geo Point - you may need to create scripted field)
- **Tooltip:** `train_id`, `speed_kmph`, `delay_min`
- **Size:** Based on `passenger_count`

---

### 🏢 Metro Station Events (`metro-station-events*`)

#### 7. **Platform Occupancy Heatmap** (Heat Map)
- **X-axis:** `station_id` (Terms)
- **Y-axis:** `ts` (Date Histogram)
- **Color:** `platform_occupancy` (Average)
- **Color gradient:** Blue (low) → Red (high)

#### 8. **Station Wait Times** (Vertical Bar Chart)
- **X-axis:** `station_id` (Terms)
- **Y-axis:** `avg_wait_min` (Average)
- **Color:** By `avg_wait_min` (gradient)
- **Sort:** Descending by wait time

#### 9. **Station Occupancy Trend** (Line Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `platform_occupancy` (Average)
- **Split by:** `station_id` (Top 5 stations)
- **Color:** Different color per station

#### 10. **Alerts by Station** (Tag Cloud)
- **Field:** `alerts.keyword` (Terms)
- **Size:** Count of documents
- **Color:** By alert type

---

### 📈 Metro KPIs (`metro-kpis*`)

#### 11. **System Overview Dashboard** (Metric Visualization)
- **Metrics to display:**
  - `trains_active` (Number format)
  - `avg_delay_min` (Number, 2 decimals)
  - `passengers_total` (Number format)
  - `on_time_percent` (Percentage, 2 decimals)
  - `crowded_stations` (Number format)

#### 12. **Average Delay Trend** (Line Chart)
- **X-axis:** `ts` (Date Histogram, 30s interval)
- **Y-axis:** `avg_delay_min` (Average)
- **Color:** Red if > 5, Green if ≤ 5
- **Time Range:** Last 15 minutes

#### 13. **On-Time Performance** (Line Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `on_time_percent` (Average)
- **Reference line:** 95% (target)
- **Color:** Green gradient

#### 14. **Total Passengers Over Time** (Area Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `passengers_total` (Average)
- **Fill:** Gradient fill
- **Color:** Blue to purple

#### 15. **Crowded Stations Count** (Bar Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `crowded_stations` (Average)
- **Color:** Red if > 3, Yellow if 1-3, Green if 0

#### 16. **Average Speed Trend** (Line Chart)
- **X-axis:** `ts` (Date Histogram)
- **Y-axis:** `avg_speed_kmph` (Average)
- **Color:** Blue gradient

---

### 🗺️ Metro Route Plans (`metro-route-plans*`)

#### 17. **Efficiency Score Distribution** (Histogram)
- **X-axis:** `efficiency_score` (Histogram, 0.1 intervals)
- **Y-axis:** Count
- **Color:** Green (0.8-1.0), Yellow (0.6-0.8), Red (0-0.6)

#### 18. **Delay Score by Train** (Bar Chart)
- **X-axis:** `train_id` (Terms)
- **Y-axis:** `delay_score` (Average)
- **Color:** By delay score (gradient)
- **Sort:** Descending

---

## 🎨 Visualization Tips for Beautiful Graphs

### Color Schemes
- **Delays:** Red (high) → Yellow (medium) → Green (low)
- **Occupancy:** Blue (low) → Purple (medium) → Red (high)
- **Performance:** Green (good) → Yellow (warning) → Red (critical)

### Time Intervals (for Date Histograms)
- **Last 15 minutes:** 30 seconds
- **Last 1 hour:** 5 minutes
- **Last 24 hours:** 1 hour

### Aggregations
- **Average:** For trends (delay, speed, occupancy)
- **Sum:** For totals (passengers)
- **Max:** For peak values
- **Min:** For minimum values
- **Count:** For frequencies

### Filtering
- Use filters to focus on:
  - Specific trains: `train_id: TRN-000`
  - Delayed trains: `delay_min > 3`
  - Crowded stations: `platform_occupancy > 350`
  - High wait times: `avg_wait_min > 5`

---

## 📋 Quick Reference: Field Names

### Train Events
- `train_id` - Train identifier
- `location` - Geo coordinates [lat, lon]
- `speed_kmph` - Speed in km/h
- `delay_min` - Delay in minutes
- `passenger_count` - Current passengers
- `capacity` - Train capacity (200)
- `next_station` - Next station ID
- `status` - Train status
- `ts` - Timestamp

### Station Events
- `station_id` - Station identifier
- `platform_occupancy` - People at platform
- `avg_wait_min` - Average wait time
- `alerts` - Array of alert strings
- `ts` - Timestamp

### KPIs
- `trains_active` - Number of active trains
- `avg_delay_min` - Average delay
- `avg_speed_kmph` - Average speed
- `passengers_total` - Total passengers
- `crowded_stations` - Count of crowded stations
- `on_time_percent` - Percentage on-time
- `ts` - Timestamp

### Route Plans
- `train_id` - Train identifier
- `planned_stops` - Array of station IDs
- `total_distance_km` - Route distance
- `delay_score` - Delay metric
- `efficiency_score` - Efficiency (0-1)
- `ts_generated` - Timestamp

---

## 🚀 Recommended Dashboard Layout

1. **Top Row:** KPI Metrics (4-6 metric visualizations)
2. **Second Row:** Delay Trend + On-Time Performance
3. **Third Row:** Passenger Load + Speed Distribution
4. **Fourth Row:** Station Occupancy Heatmap
5. **Bottom:** Map view of train locations

This creates a comprehensive real-time monitoring dashboard!

