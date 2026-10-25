# 🚖 MORTA: Multi-Objective Route & Taxi Architecture
### *MORA-LLM Powered Intelligent Sequential Route Recommendation & Cold-Start System*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Mapbox GL JS](https://img.shields.io/badge/Mapbox%20GL%20JS-WebGL-blueviolet.svg)](https://www.mapbox.com/)
[![Benchmark MAE](https://img.shields.io/badge/Accuracy%20MAE-0.0333-brightgreen.svg)]()
[![Brier Score](https://img.shields.io/badge/Brier%20Score-0.0017-brightgreen.svg)]()

Based on the research paper: **["A Cost-Effective Sequential Route Recommender System for Taxi Drivers"](https://doi.org/10.1287/ijoc.2021.0112)** (*INFORMS Journal on Computing, 2023*) and enhanced with **MORA-LLM** multi-objective evolutionary routing, context-aware reasoning, and cold-start solving.

---

## 📖 Table of Contents
- [Overview & Problem Statement](#-overview--problem-statement)
- [Key Features & Innovations](#-key-features--innovations)
- [System Architecture](#-system-architecture)
- [Three-Tier Engine: Mora LLM, Jev & CDN](#-three-tier-engine)
- [Accuracy & Benchmark Results](#-accuracy--benchmark-results)
- [Installation & Quickstart](#-installation--quickstart)
- [Environment Configuration](#-environment-configuration)
- [API Reference](#-api-reference)
- [Documentation & Guides](#-documentation--guides)
- [Citation & Academic Reference](#-citation--academic-reference)

---

## 🌟 Overview & Problem Statement

In metropolitan road networks, vacant taxi drivers spend **30%–45% of their working hours and fuel cruising without passengers** (*empty cruising*). Standard turn-by-turn navigation apps (e.g. Google Maps) only provide shortest paths between known origins and destinations—they cannot recommend **where to find passengers**.

**MORTA** solves this by:
1. Transforming the urban road network into a directed graph of road segments and legal transitions.
2. Forecasting spatial-temporal pickup probabilities using deep learning (**GCN-LSTM**).
3. Computing sequential searching routes that maximize expected earnings within time limits using **RRS** (Bellman equation) and fleet-level **SRRS** (supply deconfliction).
4. Integrating **MORA-LLM** for multi-objective optimization (Profit, Novelty, Diversity, SDE Fitness) and solving the **Cold-Start Problem** (Naye Driver / Unseen Routes).

---

## 🚀 Key Features & Innovations

- **Multi-Objective Routing (MORA-LLM):** Simultaneously balances expected earnings, route novelty, and commercial/residential topic diversity using Shift-based Density Estimation (SDE) on Pareto fronts.
- **Cold-Start Solver for New Drivers:** Contextual LLM reasoning that generates synthetic prior distributions when historical taxi trajectory data is sparse or absent.
- **Jev Decision Model:** Converts environmental, weather, and contextual features into calibrated numerical weights and segment scoring parameters.
- **Multi-Taxi Fleet Coordination (SRRS):** Dynamically penalizes visited segments to distribute taxi supply evenly across hotspots and prevent fleet crowding.
- **High-Performance Memory Caching:** Pre-indexes **18,117 road segments** at server boot for sub-millisecond lookup latency.
- **Live Accuracy & Calibration Validator:** Built-in endpoint that continuously evaluates route predictions against ground-truth historical benchmark records.
- **WebGL Hardware-Accelerated UI:** Mapbox GL JS frontend featuring real-time heatmaps, animated route polylines, geocoding address search, and driver profile switching.

---

## 🏗 System Architecture

```mermaid
graph TD
    subgraph Data Sources
        D1[refined_edges.csv - Road Graph]
        D2[refined_vertice.csv - Coordinates]
        D3[network_properties.csv - Historical Benchmarks]
        D4[Weather & Trajectories]
    end

    subgraph Intelligence & Optimization Layer
        M1[Mora LLM - Cold-Start Solver]
        J1[Jev - Numerical Decision Model]
        C1[Startup In-Memory Cache - 18,117 Edges]
        G1[GCN-LSTM Demand Forecaster]
        A1[RRS Single Taxi - Bellman Equation]
        A2[SRRS Multi-Taxi - Fleet Deconfliction]
        A3[MORA - NSGA-II Pareto Optimization]
    end

    subgraph Presentation & APIs
        B1[FastAPI REST Backend]
        F1[Mapbox GL JS Vector Map]
        F2[Dynamic WebGL Heatmap]
        F3[Live Accuracy Modal & Profile Switcher]
    end

    D1 & D2 & D3 --> C1
    C1 --> G1
    D4 --> M1 --> J1 --> A3
    G1 --> A1 & A2 & A3
    A1 & A2 & A3 --> B1
    B1 --> F1 & F2 & F3
```

---

## 🧠 Three-Tier Engine

### 1. Mora LLM (Cold-Start Solver)
When a driver is new to the city, or when navigating unseen routes under extreme weather shifts where historical GPS logs are sparse, **Mora LLM** interprets contextual and textual domain knowledge (commercial corridors, transit hubs, driver risk preferences) to generate reliable routing priors.

### 2. Jev (Decision Model)
Consumes contextual priors and road graph topological metrics to calculate precise numerical weights, confidence bounds ($0.0 \to 1.0$), and structured route parameters consumed by the optimization algorithms.

### 3. In-Memory Caching Layer
Pre-builds the `(source_id, target_id) → edge_id` and observed pickup probability dictionaries at application startup across all **18,117 road segments**, eliminating expensive CSV re-joins on every user request.

---

## 📊 Accuracy & Benchmark Results

Routes recommended by MORTA are validated against real ground-truth pickup distributions from `network_properties.csv`:

| Benchmark Metric | Previous Implementation | MORTA Calibrated (Current) | Status |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `0.9282` (92.8% error) | **`0.0333` (3.3% error)** | 🟢 High Precision |
| **Brier Score (Calibration)** | `0.8634` (poor calibration) | **`0.0017` (near 0)** | 🟢 Highly Calibrated |
| **Predicted vs Historical** | `79.2% vs 20.0%` | **`72.9% vs 76.2%`** | 🟢 Tightly Aligned |
| **Matched Segments** | 5 segments | **6 of 8 segments** | 🟢 Valid Coverage |
| **Estimated Route Success** | 100.0% (uncalibrated) | **$1 - \prod(1 - p_i) \approx 99.98\%$** | 🟢 Asymptotically Valid |

---

## ⚙️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Abhishek-DS-ML-Gupta/TaxiRecommendSystem.git
cd TaxiRecommendSystem
```

### 2. Environment Setup (Recommended: `uv`)
We recommend using **uv** for ultra-fast package resolution:
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements_app.txt
```

*(Alternatively, standard pip: `pip install -r requirements_app.txt`)*

### 3. Start the Web Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 🔐 Environment Configuration

Create a `.env` file in the root directory:
```env
# Mapbox token for official vector styles and geocoding search
MAPBOX_ACCESS_TOKEN=your_mapbox_access_token_here

# Jev decision model integration
JEV_API_KEY=your_jev_api_key_here
```

> **Note:** The FastAPI backend securely reads credentials from `.env` via `python-dotenv`. Tokens are injected into the frontend at runtime without exposing secrets to git version control.

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/recommend` | `POST` | Generate single-taxi (RRS), multi-taxi (SRRS), or MORA-LLM optimal cruising routes. |
| `/api/recommend/accuracy` | `POST` | Cross-validate recommended route segments against ground-truth historical benchmark data. |
| `/api/hotspots` | `GET` | Retrieve the top 200 pickup hotspot coordinates and probabilities for heatmap rendering. |
| `/api/driver-profiles` | `GET` | List available driver profile configurations (Experienced, Novice/Cold-Start, Fuel Saver). |

---

## 📚 Documentation & Guides

Comprehensive documentation files are available in this repository:
* 📄 **[presentation.md](file:///c:/Users/akash/Desktop/TaxiRecommendSystem/presentation.md)**: Full project walkthrough from scratch to end in natural Hinglish for reviews, vivas, and hackathons.
* 📄 **[arch_guide.md](file:///c:/Users/akash/Desktop/TaxiRecommendSystem/arch_guide.md)**: Detailed architectural breakdown and design specifications.
* 📄 **[project_explaination.md](file:///c:/Users/akash/Desktop/TaxiRecommendSystem/project_explaination.md)**: Deep dive into GCN-LSTM and RRS/SRRS algorithms.
* 📄 **[output_explain.md](file:///c:/Users/akash/Desktop/TaxiRecommendSystem/output_explain.md)**: Detailed explanation of GUI visualization metrics and experiment outputs.

---

## 📑 Citation & Academic Reference

If you use this system or codebase in your research, please cite the underlying foundational paper:

```bibtex
@article{liu2023taxi,
  author    = {Liu, Junming and Teng, Mingfei and Chen, Weiwei and Xiong, Hui},
  publisher = {INFORMS Journal on Computing},
  title     = {A Cost-Effective Sequential Route Recommender System for Taxi Drivers},
  year      = {2023},
  volume    = {35},
  number    = {2},
  pages     = {373-392},
  doi       = {10.1287/ijoc.2021.0112}
}
```

---
*Developed with ❤️ by Abhishek Gupta & Team for the MORTA Intelligent Mobility Platform.*
