# 🚖 Architecture + Input Guide — Taxi Route Recommender

---

## 🔑 API Keys Configured (`.env`)

| Key | Purpose |
|-----|---------|
| `MAPBOX_ACCESS_TOKEN` | Powers the interactive map, heatmap, geocoding search (address → Beijing map) |
| `JEV_TOKEN` | JEV Decision Model — route scoring, pickup probability weighting, confidence calculation |

Both are **injected at runtime** by FastAPI → never exposed to the frontend directly.

---

## 📂 Data Pipeline (What feeds the system)

### `refined_edges.csv` — Road Segments (38,723 edges)
```
Format: edge_id, source_node_id, target_node_id
Example: 113959, 44927, 44928
```
- Represents **directed road segments** of Beijing's road network
- 13,287 intersections (nodes), 38,723 directed road links (edges)
- **Used by:** NetworkX graph builder → routing graph → RRS/SRRS/MORA path search

---

### `refined_vertice.csv` — Intersection Coordinates (13,287 nodes)
```
Format: node_id, latitude, longitude
Example: 44927, 39.889351, 116.33559
```
- Maps every **intersection node → GPS coordinate**
- Bounding box: **Lat 39.856–39.970 | Lon 116.305–116.465** (Central + SW Beijing)
- **Used by:** Computing segment midpoints → snapping click to nearest road segment

---

### `network_properties.csv` — Pickup Probabilities & Earnings (18,301 rows)
```
Format: source_id, target_id, src_lat, src_lon, tgt_lat, tgt_lon,
        pickup_count, taxi_count, pickup_prob, expected_earning
Example: 89473, 89474, 39.9487, 116.3835, 39.9488, 116.3830,  9, 4, 0.307, 40.64
```
- **pickup_prob** = historical pickup rate for each road segment
- **expected_earning** = average ¥ fare earned from that segment
- Average pickup_prob: **0.474** | Average earning: **¥36.9**
- **Used by:** GCN-LSTM predictor → segment scoring → algorithm input

---

### `sample_trajectory.csv` — Taxi GPS Trajectories
```
Format: longitude, latitude, speed, heading, vacant_flag, passenger_flag, timestamp
Example: 116.1037, 39.7136, 102.0, 186, 0, 1, "2013-12-01 00:00:04"
```
- Records actual taxi GPS pings every ~10 seconds
- **vacant_flag=0, passenger_flag=1** → taxi has passenger (pickup happened)
- **Used by:** Training the GCN-LSTM model (historical pickup events)

---

### `weather.csv` — Hourly Weather Data
```
Format: apparentTemperature, cloudCover, datetime, humidity, icon,
        precipIntensity, precipProbability, pressure, temperature, windSpeed ...
```
- Covers Dec 2013 – Feb 2014 (Beijing winter dataset)
- **Used by:** Scenario modifier — Rainy scenario boosts pickup_prob × 1.6

---

## 🏗️ Three-Layer Architecture

### Layer 1 — MORA-LLM: Cold-Start Solver
**Problem it solves:** New driver with no trip history → what route to take?

```
Input: driver_persona + start_position + pickup_probs + scenario
  ↓
LEPS (LLM-Enhanced Persona Synthesis):
  - Reads driver profile (new/veteran/night/airport/eco)
  - JEV scores each candidate route using numerical weights
  - LLM generates natural language reasoning
  ↓
Multi-Objective Evolutionary Search (NSGA-II style):
  - Population of 25 candidate routes, 20 generations
  - 3 objectives: f1=Profit, f2=Novelty, f3=Diversity
  - SDE (Shift-based Density Estimation) for Pareto selection
  ↓
Output: Pareto-optimal route + LLM explanation + 3 alternatives
        (Balanced / Max Profit / Max Novelty)
```

| Driver Persona | Cold Start? | JEV Weight Strategy |
|---------------|-------------|---------------------|
| Alex Zhang (New) | ❄️ Yes | Safe arterials, low-risk corridors |
| Master Wang (Veteran) | No | CBD hotspots, historical pattern |
| Sarah Liu (Night) | No | Surge zones, late-night clusters |
| Li Na (Airport) | No | Long-haul, transit hubs |
| Chen Wei (Eco) | No | High-efficiency, anti-idling |

---

### Layer 2 — JEV: Decision Model
**What it does:** Numerical scoring engine that powers all route evaluation

```
Inputs (per road segment):
  - pickup_prob        → from network_properties.csv
  - expected_earning   → from network_properties.csv
  - taxi_count         → competition on this segment
  - alpha (α)          → dynamic correction factor (Eq. 8 from paper)
  - scenario modifier  → normal×1.0, peak×1.4, rainy×1.6, night×0.6

JEV Score Formula (Bellman Equation variant):
  score(seg) = pickup_prob × expected_earning 
             + alpha × dynamic_correction
             - competition_penalty(taxi_count)

Output: ranked route scores → input to RRS/SRRS/MORA optimizer
```

**CSV parameters it produces:**
- `pickup_prob` per segment (0.0–1.0)
- `expected_earning` in ¥ yuan
- `confidence` = normalized route success probability
- `sde_fitness` for MORA multi-objective selection

---

### Layer 3 — CDN: Caching Layer
**What it caches:** Heavy precomputed data loaded ONCE at server startup

```
Server starts → load_all() runs ONCE:
  ├── NetworkX DiGraph (13,287 nodes, 38,723 edges)     → cached in app_state["G"]
  ├── Segment pickup probs dict {edge_id: float}         → cached in app_state["pickup_probs"]
  ├── Earnings dict {edge_id: float}                     → cached in app_state["earnings"]
  ├── Routing graph (segment→segment transitions)        → cached in app_state["route_G"]
  ├── Edge midpoint coordinates {edge_id: (lat,lon)}     → cached in app_state["edge_coords"]
  ├── Top 200 hotspots list (for heatmap)                → cached in app_state["top_hotspots"]
  └── MORA-LLM optimizer instance                        → cached in app_state["mora_optimizer"]

API requests → read from cache (zero reload latency)
```

Map tile caching: Mapbox CDN (global edge network) caches vector map tiles.

---

## 📍 Suggested Input Locations (Real Beijing Coordinates)

These are derived from **actual `network_properties.csv` hotspots** with the best pickup probabilities and earnings:

### 🔥 High Earning Hotspots (Best for "Max Profit" strategy)

| Location Name | Latitude | Longitude | Pickup Prob | Est. Earning |
|--------------|----------|-----------|-------------|-------------|
| 🏙️ CBD / Guomao Area | **39.9178** | **116.4506** | 23.1% | ¥90.9 |
| 🏬 Chaoyang / Workers Stadium | **39.9337** | **116.3897** | 23.5% | ¥90.0 |
| 🚉 Fengtai / South Station area | **39.8947** | **116.4129** | 25.0% | ¥93.3 |
| 🌉 North 4th Ring / Haidian | **39.9581** | **116.3486** | 27.6% | ¥91.7 |
| 🏨 Daxing / South City | **39.8788** | **116.3798** | 33.3% | ¥89.6 |

### 🎯 Balanced Hotspots (Good for MORA "Balanced" Pareto option)

| Location Name | Latitude | Longitude | Pickup Prob | Est. Earning |
|--------------|----------|-----------|-------------|-------------|
| 🏛️ Xicheng / Zhongguancun South | **39.9483** | **116.3250** | 100% (prob=1.0) | — |
| 🎓 Haidian / University District | **39.9370** | **116.3310** | 100% (prob=1.0) | — |
| 🏪 Fengtai / Liuliqiao | **39.8796** | **116.4621** | 50.0% | ¥80.6 |
| 🚇 Chaoyang / Sanlitun | **39.9092** | **116.4251** | 25.0% | ¥80.6 |
| 🛣️ North 5th Ring | **39.9669** | **116.4249** | 40.0% | ¥82.2 |

### 🗺️ How to Enter These in the App

**Option 1 — Click on the map:**
- Open http://localhost:8000
- Click directly on the map at any of the coordinates above

**Option 2 — Use the Search Bar (Mapbox Geocoding):**
```
Type any of these into the search box on the map:
  - "国贸" or "Guomao, Beijing"              → CBD area (¥90.9)
  - "工人体育场" or "Workers Stadium Beijing" → Chaoyang hotspot
  - "北京南站" or "Beijing South Station"    → Fengtai area
  - "海淀区" or "Haidian District Beijing"   → University zone
  - "三里屯" or "Sanlitun Beijing"           → Nightlife hotspot (use Night scenario)
  - "中关村" or "Zhongguancun Beijing"       → Tech hub, high prob
```

---

## 🧪 Recommended Test Workflow

```
1. Open http://localhost:8000

2. COLD-START TEST (New Driver):
   Algorithm: MORA-LLM
   Persona: Alex Zhang (Cold Start ❄️)
   Click: lat=39.9337, lon=116.3897 (Chaoyang area)
   α=0.20, τ=10min, Scenario=Normal
   → Expected: Safe arterial route, AI explains reasoning

3. VETERAN PEAK HOUR TEST:
   Algorithm: MORA-LLM
   Persona: Master Wang (Veteran)
   Click: lat=39.9178, lon=116.4506 (CBD)
   α=0.40, τ=15min, Scenario=Peak
   → Expected: High-earning CBD corridor, ¥80-90+ routes

4. MULTI-TAXI FAIRNESS TEST:
   Algorithm: SRRS
   Taxis: 3
   Click: lat=39.9483, lon=116.3250 (Haidian)
   → Expected: 3 non-overlapping routes, each to different zones

5. NIGHT SURGE TEST:
   Algorithm: MORA-LLM
   Persona: Sarah Liu (Night Shift)
   Click: lat=39.9092, lon=116.4251 (Sanlitun)
   Scenario: Night
   → Expected: Night-adapted route, surge-sensitive corridors
```

---

## 📊 Data Schema Reference

```
refined_edges.csv    → edge_id | source_node | target_node
refined_vertice.csv  → node_id | lat | lon
network_properties   → src_id | tgt_id | src_lat | src_lon | tgt_lat | tgt_lon |
                       pickup_count | taxi_count | pickup_prob | expected_earning
sample_trajectory    → lon | lat | speed | heading | vacant | passenger | timestamp
weather.csv          → temperature | humidity | icon | precipProb | windSpeed | datetime
```
