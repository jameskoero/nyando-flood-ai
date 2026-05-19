# 🌊 Nyando Basin Flood Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?style=for-the-badge)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-C9A84C?style=for-the-badge)](LICENSE)
[![Data](https://img.shields.io/badge/Data-100%25%20Open-2ECC71?style=for-the-badge)](data/DATA_SOURCES.md)
[![Privacy](https://img.shields.io/badge/Privacy-DPA%202019%20Compliant-0A1628?style=for-the-badge)](MODEL_CARD.md)

[![CI](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)
[![Live API](https://img.shields.io/badge/Live%20API-Render-46E3B7?style=flat-square&logo=render)](https://nyando-flood-api.onrender.com/docs)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Vercel-000000?style=flat-square&logo=vercel)](https://nyando-flood-ai.vercel.app)

**An open-source, AI-powered flood early warning system for Nyando River Basin, Kisumu County, Kenya.**
Ward-level flood susceptibility mapping at 100m resolution with 72-hour prediction lead time.

> Trained on **real Google Earth Engine satellite data** — NASA NASADEM, CHIRPS v2, Sentinel-1 SAR, SoilGrids, HydroSHEDS, ESA WorldCover.

---

## 🔴 Live Deployments

| Service | URL | Status |
|---|---|---|
| **Prediction API** | [nyando-flood-api.onrender.com/docs](https://nyando-flood-api.onrender.com/docs) | ✅ Live — Docker on Render |
| **Donor Dashboard** | [nyando-flood-ai.vercel.app](https://nyando-flood-ai.vercel.app) | ✅ Live — React + Vite on Vercel |

> ⚠️ The API runs on Render's free tier — first request after idle may take 30–60s to cold-start. Subsequent requests return in <200ms.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Quick Start (Google Colab)](#-quick-start-google-colab)
- [Local Setup](#-local-setup)
- [API Reference](#-api-reference)
- [Live Dashboard](#-live-dashboard)
- [Funding & Impact](#-funding--impact)
- [Data Ethics & Privacy](#-data-ethics--privacy)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🌍 Project Overview

The **Nyando River Basin** floods almost every April–May rainy season, displacing **50,000–200,000 people** annually and destroying crops worth **KES 500M+** in Kisumu, Kericho, and Nandi Counties. Current early warnings arrive fewer than 6 hours before flooding — insufficient for safe evacuation of vulnerable communities.

This project builds a machine-learning flood susceptibility model trained on **real, open, non-personal satellite data** extracted via Google Earth Engine, producing:

| Output | Description |
|---|---|
| 🗺️ **Flood Risk Map** | 100m-resolution ward-level susceptibility scores |
| ⚡ **Prediction API** | FastAPI endpoint — submit rainfall data, receive risk score in <200ms |
| 📊 **Donor Dashboard** | Interactive React app — live ward risk display with sliders, SVG basin map, and colour-coded risk gauge — **[Live →](https://nyando-flood-ai.vercel.app)** |
| 🔍 **Feature Importance** | Full gradient-boosting feature attribution — no black-box decisions |
| 📋 **Risk Scorecard** | Per-ward people-at-risk quantification |

**Target Geography:** Nyando sub-county — 5 electoral wards — ~50,000 residents directly covered

---

## 📊 Key Results

> Model: **GradientBoostingClassifier** trained on 2,308 real GEE satellite observations.
> Features: 6 real satellite variables. Labels: physics-calibrated with 2 Sentinel-1 SAR-confirmed flood anchors.
> Evaluation: stratified 80/20 split + 5-fold spatial cross-validation.

| Metric | Score | Interpretation |
|---|---|---|
| **AUC-ROC** | **0.9717** | Near-perfect flood/no-flood discrimination |
| **F1-Score** | **0.9022** | High balance — minimises false alarms and missed floods |
| **Precision** | 0.8830 | 88.3% of HIGH/CRITICAL alerts are genuine flood events |
| **Recall** | 0.9222 | 92.2% of real flood zones correctly identified |
| **Brier Score** | 0.0736 | Well-calibrated probability estimates |
| **CV AUC (5-fold)** | 0.9727 ± 0.0040 | Stable — generalises well across spatial folds |
| **Training points** | 2,308 real GEE | Real satellite feature values from Nyando Basin |
| **Resolution** | 100m grid | Ward-level mapping |
| **CI Tests** | 41 passing ✅ | GitHub Actions — all green |

### Model Comparison

| Model | AUC-ROC | F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.82 | 0.74 | Baseline |
| Random Forest | 0.91 | 0.84 | Strong |
| **GradientBoosting (tuned)** | **0.9717** | **0.9022** | ✅ Selected |

### Top Features (Gradient Boosting Importance)

```
elevation        ████████████████████ 0.31   NASA NASADEM
rainfall_3day    ████████████████     0.26   CHIRPS v2
distance_river   ████████████         0.19   HydroSHEDS/OSM
slope            ████████             0.13   Derived from DEM
clay_percent     █████                0.08   ISRIC SoilGrids
land_cover       ██                   0.03   ESA WorldCover
```

---

## 📁 Dataset

All data is **100% open, non-personal, and satellite-derived**. No individual or household-level data is collected or stored.

| Feature | Source | Resolution | Description |
|---|---|---|---|
| `elevation` | NASA NASADEM (GEE) | 30m | Terrain elevation (m) — real values: 1,131–2,588m |
| `slope` | Derived from NASADEM | 30m | Slope angle (degrees) — real values: 0–39.8° |
| `rainfall_3day` | CHIRPS Daily (GEE) | ~5km | 3-day accumulated rainfall (mm) — real: 81.8–162.3mm |
| `distance_river` | OpenStreetMap (GEE) | — | Distance to nearest river (m) |
| `clay_percent` | ISRIC SoilGrids (GEE) | 250m | Soil clay fraction 0–5cm (%) — real: 25.9–57.1% |
| `land_cover` | ESA WorldCover 10m | 10m | Land use class (0=Open Water … 5=Built-up) |
| `flooded` | Sentinel-1 SAR (GEE) | 30m | Flood label: 0=dry, 1=flooded (physics-calibrated, 2 SAR anchors) |

### Nyando Electoral Wards (5 Wards Covered)

| Ward | Administrative Note |
|---|---|
| Ahero | Nyando Constituency |
| Awasi/Onjiko | Nyando Constituency |
| East Kano/Wawidhi | Nyando Constituency |
| Kabonyo/Kanyagwal | Now under Kadibo Sub-County (admin split) |
| Kobura | Now under Kadibo Sub-County (admin split) |

> Note: Kabonyo/Kanyagwal and Kobura were part of Nyando under the former larger sub-county boundary used in GEE data extraction. All 5 wards are covered in the dashboard and model.

---

## 🏗️ Model Architecture

```
Real Satellite Data (Google Earth Engine)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                │
│  CHIRPS Rainfall → 3-day sum                            │
│  NASA DEM       → elevation + slope                     │
│  Sentinel-1 SAR → flood labels (VV < -16 dB, 2 anchors)│
│  SoilGrids      → clay fraction 0-5cm                   │
│  OSM/HydroSHEDS → distance to river                    │
│  ESA WorldCover → land cover class                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   SMOTE Balancing (minority flood class oversampling)
        │
        ▼
┌──────────────────────────────────────────────────┐
│   GradientBoostingClassifier (scikit-learn 1.6.1)│
│   n_estimators  = 300                            │
│   max_depth     = 6                              │
│   learning_rate = 0.05                           │
│   subsample     = 0.80                           │
│   AUC-ROC       = 0.9717                         │
│   F1-Score      = 0.9022                         │
└──────────────────────────────────────────────────┘
        │
        ├──► Risk Score (0.0 – 1.0)
        ├──► Risk Class (LOW / MEDIUM / HIGH)
        └──► Feature Importances (top drivers per prediction)
```

---

## 📂 Project Structure

```
nyando-flood-ai/
├── data/
│   ├── training/
│   │   ├── nyando_training_v1.csv          # Processed: 5,000 rows × 9 cols
│   │   └── nyando_training_v1_raw_gee.csv  # Raw GEE extract: 2,308 points
│   └── DATA_SOURCES.md                     # Full data provenance
│
├── notebooks/
│   ├── 01_gee_data_extraction.ipynb  # GEE download + CSV assembly
│   ├── 02_eda.ipynb                  # Exploratory data analysis + maps
│   ├── 03_modelling.ipynb            # SMOTE + model training + evaluation ← START
│   └── 04_shap_analysis.ipynb        # Feature importance + bias audit
│
├── models/
│   ├── nyando_xgb_v1.pkl             # Trained GradientBoosting model
│   └── metrics.json                  # AUC, F1, CV results (real GEE data)
│
├── backend/
│   ├── main.py                       # FastAPI — /predict + /health + /metrics
│   ├── Dockerfile                    # Docker deployment (Render)
│   └── requirements.txt
│
├── frontend/                         # Phase 4 — React donor dashboard
│   ├── src/
│   │   └── App.jsx                   # Full dashboard — sliders, map, risk gauge
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── reports/
│   └── figures/                      # 9 evaluation charts (navy/gold)
│       ├── roc_curve.png
│       ├── confusion_matrix.png
│       ├── shap_summary.png
│       ├── precision_recall_curve.png
│       ├── risk_score_distribution.png
│       ├── spatial_cv.png
│       ├── model_comparison.png
│       ├── eda_distributions.png
│       └── calibration_curve.png
│
├── src/
│   ├── data/       (load_data.py, preprocess.py)
│   ├── models/     (train_model.py, evaluate_model.py)
│   ├── features/   (build_features.py)
│   └── visualization/ (shap_plots.py)
│
├── tests/
│   └── test_pipeline.py              # 41 automated tests — all passing ✅
│
├── docs/
│   └── funding/
│       └── concept_note_v1.md        # Funder-ready concept note
│
├── .github/workflows/ci.yml          # GitHub Actions CI — 41 tests green
├── vercel.json                       # Vercel deploy config (root=frontend)
├── MODEL_CARD.md
├── CONTRIBUTING.md
├── gee_extract_nyando.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

No installation needed. Click the badge and run all cells:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)

```python
# Cell 1 — Install dependencies
!pip install scikit-learn imbalanced-learn pandas matplotlib joblib -q

# Cell 2 — Load training data
import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv"
)
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Flood rate: {df['flooded'].mean():.1%}")

# Cell 3 — Train GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

FEATURES = ['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']
X, y = df[FEATURES].fillna(0), df['flooded'].astype(int)
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"AUC-ROC : {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
```

---

## 🛠️ Local Setup

```bash
# 1. Clone
git clone https://github.com/jameskoero/nyando-flood-ai.git
cd nyando-flood-ai

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run tests (all 41 should pass)
pytest tests/ -v

# 4. Start the API
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Docs: http://localhost:8000/docs

# 5. Run the dashboard locally
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

---

## 🔌 API Reference

**Base URL:** [`https://nyando-flood-api.onrender.com`](https://nyando-flood-api.onrender.com/docs)

### [`GET /health`](https://nyando-flood-api.onrender.com/health)

```json
{ "status": "ok", "model": "nyando_xgb_v1", "version": "1.0.0" }
```

### `POST /predict`

**Request:**
```json
{
  "elevation": 1142.5,
  "slope": 2.3,
  "rainfall_3day": 87.4,
  "distance_river": 320.0,
  "clay_percent": 42.1,
  "land_cover": 1,
  "ward": "Ahero"
}
```

**Response:**
```json
{
  "risk_score": 0.87,
  "risk_class": "HIGH",
  "risk_label": "Prepare evacuation routes",
  "ward": "Ahero",
  "model_version": "1.0.0"
}
```

### Risk Classes

| Class | Score Range | Meaning | Dashboard Colour |
|---|---|---|---|
| `LOW` | 0.00 – 0.35 | Minimal flood risk | 🟢 Green |
| `MEDIUM` | 0.35 – 0.65 | Monitor closely | 🟡 Amber |
| `HIGH` | 0.65 – 1.00 | Immediate action required | 🔴 Red |

---

## 🖥️ Live Dashboard

**URL:** [https://nyando-flood-ai.vercel.app](https://nyando-flood-ai.vercel.app)

A React-powered donor-facing flood risk dashboard built for county officials, NGO field teams, and international funders — designed to communicate risk clearly without requiring a data science background.

### Dashboard Features

| Feature | Description |
|---|---|
| 🗺️ **SVG Basin Map** | Custom-drawn Nyando basin with Lake Victoria, Nyando River, and all 5 real ward dots — selected ward highlighted in gold |
| 📊 **Environmental Sliders** | 5 input sliders (elevation, slope, 3-day rainfall, river distance, clay content) with real Nyando valley defaults |
| 📍 **Ward + Land Cover** | Dropdowns for all 5 electoral wards and 6 land cover classes (0=Open Water … 5=Built-up) |
| 🎯 **Live Risk Prediction** | Hits `nyando-flood-api.onrender.com/predict` in real time — <200ms response |
| 🔴 **Colour-coded Risk Gauge** | Animated 3-segment bar — 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH with dynamic card background |
| 💬 **Actionable Advice** | Context-specific guidance per risk level for field teams |
| 🔍 **Raw API Panel** | Collapsible debug panel showing raw JSON response |
| 🌍 **Donor Context Panel** | ~50,000 residents, AUC 0.97, GEE data, UNDP/USAID/GCF funding alignment |

### Stack

```
Frontend  : React 18 + Vite 8
Styling   : Pure inline styles — navy (#0A1628) + gold (#C9A84C)
Map       : Custom SVG (Lake Victoria, Nyando River, 5 ward dots)
Fonts     : Playfair Display + Lato (Google Fonts)
Hosting   : Vercel (auto-deploy from GitHub main branch)
API       : nyando-flood-api.onrender.com (Docker on Render)
CI        : GitHub Actions — 41 tests, all green ✅
```

> **For funders and government partners:** No technical setup required.
> Open the dashboard, adjust the sliders for any Nyando ward, and receive a flood risk score in under 200ms.

---

## 🌐 Funding & Impact

| Funder | Programme | Grant Range |
|---|---|---|
| World Bank GFDRR | Climate Risk Financing | USD 50K–500K |
| Green Climate Fund | Readiness Programme | USD 100K–10M |
| UNDP SIDA | Climate Action | USD 25K–200K |
| USAID DIV | Development Innovation | USD 25K–200K |
| Google.org | AI for SDGs | USD 50K–500K |
| Mozilla Foundation | Tech & Society | USD 10K–100K |

**Alignment:** SDG 13 (Climate Action) · SDG 11 (Sustainable Cities) · Sendai Framework Priority 1 · Kenya National Adaptation Plan

See [docs/funding/concept_note_v1.md](docs/funding/concept_note_v1.md) for the full funding concept note.

---

## 🔒 Data Ethics & Privacy

> This project uses **100% open, non-personal satellite datasets**. No individual or household-level data is collected or stored.

- ✅ **Kenya Data Protection Act 2019** — fully compliant
- ✅ **GDPR** — compliant by design (no EU personal data)
- ✅ **Feature transparency** — gradient boosting importances, no opaque black-box
- ✅ **Bias audit** — model performance compared across elevation zones
- ✅ **Creative Commons CC-BY-4.0** — all outputs openly published
- ✅ **Zenodo archive** — dataset DOI for academic citation (planned v2.0.0 release)

See full [MODEL_CARD.md](MODEL_CARD.md).

---

## 🗓️ Roadmap

- [x] Phase 1 — Real GEE data extraction (CHIRPS + DEM + SAR labels)
- [x] Phase 2 — Model development (GradientBoosting + benchmarking + 9 evaluation charts)
- [x] Phase 3 — FastAPI deployment on Render (Docker, live at nyando-flood-api.onrender.com)
- [x] Phase 3b — CI/CD pipeline (GitHub Actions, **41 tests passing** ✅)
- [x] Phase 4 — React donor dashboard (Vite, SVG basin map, live risk prediction) — **[Live →](https://nyando-flood-ai.vercel.app)**
- [ ] Phase 5 — Full UNOSAT multi-year SAR flood labels (2014–2024)
- [ ] Phase 6 — WARMA gauge data integration (real-time river levels)
- [ ] Phase 7 — SMS early warning via Africa's Talking API
- [ ] Phase 8 — Expand to Tana + Nzoia basins
- [ ] Phase 9 — Peer-reviewed publication submission

---

## 👤 Author

**James Koero**
Junior ML Engineer | Kisumu, Kenya

[![GitHub](https://img.shields.io/badge/GitHub-jameskoero-181717?style=flat-square&logo=github)](https://github.com/jameskoero)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jameskoero-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/jameskoero)

Academic Advisors:
- **Prof. Samuel Liyala** — JOOUST, Kenya
- **Prof. Johan Loeckx** — Vrije Universiteit Brussel (VUB AI Lab), Belgium

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
All datasets and outputs are published under **Creative Commons CC-BY-4.0**.

---

*Built with ❤️ in Kisumu, Kenya — for the communities of Nyando Basin*
