# AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

> **Smart India Hackathon (SIH) Project**

---

## Problem Statement

NASA satellites (MODIS / VIIRS) scan Earth every few hours and flag locations
with unusually high surface temperatures as "hotspots." The raw data tells us
**"heat was detected here"** — but not what caused it.

A forest wildfire, an industrial gas flare, and post-harvest stubble burning
all produce similar thermal signatures. This creates:
- **False alarms** — emergency responders dispatched to industrial sites
- **Missed real emergencies** — wildfires dismissed as known industry

**Goal:** Build an AI system that automatically classifies every satellite
hotspot into one of four categories and displays results on a live dashboard.

| Category | Colour | Description |
|---|---|---|
| 🔴 Wildfire Risk | Red | Sudden, high-intensity, unexpected fire |
| 🟠 Agricultural Burning | Orange | Seasonal farmland stubble burning |
| ⚫ Industrial (Normal) | Grey | Expected heat from known industrial activity |
| 🟡 Anomaly/Unclassified | Yellow | Doesn't fit a clear pattern — needs review |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│  NASA FIRMS API          OpenStreetMap (via OSMnx)              │
│  ─────────────           ──────────────────────────             │
│  Real-time hotspots      Land-use polygons                      │
│  lat / lon / FRP /       (industrial, forest,                   │
│  brightness / date       farmland, residential)                 │
└────────────┬──────────────────────┬────────────────────────────┘
             │                      │
             ▼                      ▼
┌────────────────────┐   ┌──────────────────────────┐
│  STEP 1            │   │  STEP 2                  │
│  fetch_firms_      │──▶│  fetch_osm_landuse.py    │
│  data.py           │   │                          │
│                    │   │  • land_use_type          │
│  firms_raw.csv     │   │  • distance_to_industrial │
└────────────────────┘   └──────────────┬───────────┘
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │  STEP 3                  │
                         │  build_features.py       │
                         │                          │
                         │  • historical_frequency  │
                         │  • time_of_day / season  │
                         │  • heuristic category    │
                         │                          │
                         │  features_labeled.csv    │
                         └──────────────┬───────────┘
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │  STEP 4                  │
                         │  train_model.py          │
                         │                          │
                         │  XGBoost multi-class     │
                         │  + SHAP explainability   │
                         │                          │
                         │  fire_classifier.pkl     │
                         └──────────────┬───────────┘
                                        │
                                        ▼
                         ┌──────────────────────────┐
                         │  STEP 5                  │
                         │  app/dashboard.py        │
                         │                          │
                         │  Streamlit + Folium map  │
                         │  Plotly trend chart      │
                         │  Wildfire alerts panel   │
                         └──────────────────────────┘
```

---

## Project Structure

```
SIH/
├── src/
│   ├── fetch_firms_data.py      # Step 1 — fetch NASA FIRMS hotspots
│   ├── fetch_osm_landuse.py     # Step 2 — OSM land-use enrichment
│   ├── build_features.py        # Step 3 — feature engineering + labelling
│   └── train_model.py           # Step 4 — XGBoost + SHAP
├── app/
│   └── dashboard.py             # Step 5 — Streamlit dashboard
├── data/                        # auto-generated (git-ignored)
│   ├── firms_raw.csv
│   ├── hotspots_with_landuse.csv
│   ├── features_labeled.csv
│   └── osm_cache/               # OSMnx tile cache
├── models/                      # auto-generated (git-ignored)
│   ├── fire_classifier.pkl
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── model_report.txt
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites

| Tool   | Minimum version |
|--------|----------------|
| Python | 3.9            |
| pip    | 23             |
| Git    | any            |

### 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd SIH
```

### 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note for geopandas:** If the install fails, try:
> ```
> pip install pipwin
> pipwin install gdal fiona pyproj shapely
> pip install geopandas
> ```

### 4 — Get a free NASA FIRMS API key

1. Go to: https://firms.modaps.eosdis.nasa.gov/api/map_key
2. Register / log in with a NASA Earthdata account (free)
3. Copy your **MAP_KEY**

### 5 — Configure environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env`:

```
FIRMS_MAP_KEY=your_actual_key_here
```

---

## Running the Pipeline

Run **one step at a time** and verify the output before continuing.

```bash
# Step 1 — Fetch satellite hotspot data for India (~seconds)
python -X utf8 src/fetch_firms_data.py

# Step 2 — Enrich hotspots with OSM land-use context (~minutes, cached)
python -X utf8 src/fetch_osm_landuse.py

# Step 3 — Feature engineering + heuristic labels (~seconds)
python -X utf8 src/build_features.py

# Step 4 — Train XGBoost classifier + SHAP plots (~1-2 minutes)
python -X utf8 src/train_model.py

# Step 5 — Launch dashboard
streamlit run app/dashboard.py
```

The dashboard opens at **http://localhost:8501**

---

## Features Explained

| Feature | Source | Description |
|---|---|---|
| `brightness` | FIRMS | Brightness temperature in Kelvin (fire indicator) |
| `frp` | FIRMS | Fire Radiative Power in MW (fire intensity) |
| `land_use_type` | OSM | industrial / forest / farmland / residential / unknown |
| `distance_to_industrial` | OSM | Metres to nearest industrial OSM geometry |
| `historical_frequency` | Computed | Times this ~1 km location fired in dataset window |
| `time_of_day` | Computed | Night / Morning / Afternoon / Evening |
| `season` | Computed | Winter / Summer / Monsoon / Post-Monsoon |

### Heuristic Labelling Rules

```
if land_use == "industrial" AND historical_frequency >= 3:
    → Industrial (Normal)

elif frp >= 50 MW AND historical_frequency < 3 AND land is forest/farmland:
    → Wildfire Risk

elif land_use == "farmland" AND month in {Oct, Nov} AND freq >= 2:
    → Agricultural Burning

else:
    → Anomaly/Unclassified
```

---

## Dashboard Features

| Tab | Content |
|---|---|
| 🗺️ Live Map | Folium map — clustered markers, colour-coded, click for details |
| 📈 Trend Chart | Daily hotspot frequency by category over time |
| 🚨 Alerts | Top 10 Wildfire Risk detections sorted by FRP intensity |
| 🤖 Model Info | Classification report, confusion matrix, SHAP importance |

---

## Deploying to Streamlit Community Cloud

See the **Deployment** section below.

---

## Data Notes

- FIRMS NRT data has ~3 hour latency after satellite overpass
- The API allows up to 10 days per request
- OSMnx caches tile responses in `data/osm_cache/` — re-runs are instant
- For historical data beyond 10 days, use the
  [FIRMS Archive portal](https://firms.modaps.eosdis.nasa.gov/download/)

---

## Roadmap

- [x] Step 1 — FIRMS data fetch
- [x] Step 2 — OSM land-use enrichment
- [x] Step 3 — Feature engineering + heuristic labelling
- [x] Step 4 — XGBoost classifier + SHAP explainability
- [x] Step 5 — Streamlit + Folium + Plotly dashboard
- [ ] Ground-truth annotation and model fine-tuning
- [ ] Scheduled auto-refresh (re-fetch FIRMS daily)
- [ ] Streamlit Cloud deployment
- [ ] Email / SMS alerts for Wildfire Risk detections
