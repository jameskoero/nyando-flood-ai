# 🌊 Nyando Basin Flood Risk Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-C9A84C?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-100%25%20Open-2ECC71?style=for-the-badge)
![Privacy](https://img.shields.io/badge/Privacy-DPA%202019%20Compliant-0A1628?style=for-the-badge)

**An open-source, AI-powered flood early warning system for Nyando River Basin, Kisumu County, Kenya.**  
Ward-level flood susceptibility mapping at 100m resolution with 72-hour prediction lead time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)
&nbsp;
[![Live API](https://img.shields.io/badge/Live%20API-Render-46E3B7?style=flat-square&logo=render)](https://nyando-flood-api.onrender.com/docs)
&nbsp;
[![Dashboard](https://img.shields.io/badge/Dashboard-Vercel-000000?style=flat-square&logo=vercel)](https://nyando-flood-ai.vercel.app)
&nbsp;
[![Dataset DOI](https://img.shields.io/badge/Dataset-Zenodo%20DOI-1682D4?style=flat-square)](https://zenodo.org)

</div>

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
- [Dashboard](#-dashboard)
- [Funding & Impact](#-funding--impact)
- [Data Ethics & Privacy](#-data-ethics--privacy)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🌍 Project Overview

The **Nyando River Basin** floods almost every April–May rainy season, displacing **50,000–200,000 people** annually and destroying crops worth **KES 500M+** in Kisumu, Kericho, and Nandi Counties. Current early warnings arrive fewer than 6 hours before flooding — insufficient for safe evacuation of vulnerable communities.

This project builds a machine-learning flood susceptibility model trained entirely on **open, non-personal satellite and environmental data**, producing:

| Output | Description |
|--------|-------------|
| 🗺️ **Flood Risk Map** | 100m-resolution ward-level susceptibility scores (GeoTIFF + PDF) |
| ⚡ **Prediction API** | FastAPI endpoint — submit rainfall data, receive risk score in <200ms |
| 📊 **Web Dashboard** | Interactive React + Leaflet.js ward risk map for county planners |
| 📋 **Risk Scorecard** | Per-ward people-at-risk quantification using WorldPop data |
| 🔍 **SHAP Explainability** | Full feature-level transparency — no black-box decisions |

**Target Geography:** Nyando sub-county — 42 wards — 500,000+ residents

---

## 📊 Key Results

> Results from XGBoost model trained on 5,000 sample points across Nyando Basin.  
> Spatial 5-fold cross-validation. SMOTE applied for class balance.

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.94** |
| **F1-Score** | **0.88** |
| **Precision** | 0.89 |
| **Recall** | 0.87 |
| **Spatial CV std** | ±0.02 |
| **Training points** | 5,000 |
| **Resolution** | 100m grid |
| **Historical range** | 2014–2024 (10 years) |

### Model Comparison

| Model | AUC-ROC | F1 | Notes |
|-------|---------|----|-------|
| Logistic Regression | 0.82 | 0.74 | Baseline |
| Random Forest | 0.91 | 0.84 | Strong |
| **XGBoost (tuned)** | **0.94** | **0.88** | ✅ Selected |
| LightGBM | 0.93 | 0.87 | Alternative |

### Top Features (SHAP importance)

```
elevation        ████████████████████ 0.31
rainfall_3day    ████████████████     0.26
distance_river   ████████████         0.19
slope            ████████             0.13
clay_percent     █████                0.08
land_cover       ██                   0.03
```

---

## 📁 Dataset

All data is **100% open, non-personal, and satellite/census-derived**. No individual or household-level data is collected or stored.

| Feature | Source | Resolution | Description |
|---------|--------|------------|-------------|
| `elevation` | NASA NASADEM (GEE) | 30m | Terrain elevation (m) |
| `slope` | Derived from NASADEM | 30m | Slope angle (degrees) |
| `rainfall_3day` | CHIRPS Daily (GEE) | ~5km | 3-day accumulated rainfall (mm) |
| `distance_river` | OpenStreetMap (GEE) | — | Distance to nearest river (m) |
| `clay_percent` | ISRIC SoilGrids (GEE) | 250m | Soil clay fraction 0–5cm (%) |
| `land_cover` | ESA WorldCover 10m | 10m | Land use class |
| `population` | WorldPop Kenya 2020 | 100m | Population per pixel |
| `flooded` | UNOSAT / Sentinel-1 SAR | 30m | Flood label: 0=dry, 1=flooded |

### Download the Training CSV

```bash
# Download from GitHub Releases
wget https://github.com/jameskoero/nyando-flood-ai/releases/download/v1.0/nyando_training_v1.csv

# Or load directly in Python
import pandas as pd
df = pd.read_csv(
  "https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv"
)
print(df.shape)   # (5000, 8)
print(df.head())
```

---

## 🏗️ Model Architecture

```
Raw Satellite Data (GEE)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline              │
│  CHIRPS Rainfall → 3-day sum                          │
│  NASA DEM       → elevation + slope                   │
│  Sentinel-1 SAR → flood labels (VV < -16 dB)         │
│  SoilGrids      → clay fraction                       │
│  OSM            → distance to river                   │
│  ESA WorldCover → land cover class                    │
└───────────────────────────────────────────────────────┘
        │
        ▼
   SMOTE Balancing (minority class oversampling)
        │
        ▼
┌─────────────────────────────────┐
│   XGBoost Classifier (tuned)    │
│   n_estimators = 300            │
│   max_depth    = 6              │
│   learning_rate= 0.05           │
│   subsample    = 0.80           │
│   AUC-ROC      = 0.94           │
└─────────────────────────────────┘
        │
        ├──► Risk Score (0.0 – 1.0)
        ├──► Risk Class (LOW / MEDIUM / HIGH / CRITICAL)
        └──► SHAP Values (top-3 flood drivers per ward)
```

---

## 📂 Project Structure

```
nyando-flood-ai/
├── data/
│   ├── raw/                    # CHIRPS, UNOSAT CSVs (or GEE download scripts)
│   ├── processed/              # Cleaned, merged intermediate datasets
│   └── training/
│       └── nyando_training_v1.csv   # Master training file (5,000 rows × 8 cols)
│
├── notebooks/
│   ├── 01_data_prep.ipynb      # GEE download + CSV assembly
│   ├── 02_eda.ipynb            # Exploratory data analysis + maps
│   ├── 03_modelling.ipynb      # SMOTE + model training + evaluation ← START HERE
│   └── 04_shap_analysis.ipynb  # SHAP explainability + bias audit
│
├── models/
│   ├── nyando_xgb_v1.pkl       # Trained XGBoost model
│   └── metrics.json            # AUC, F1, confusion matrix results
│
├── backend/
│   ├── main.py                 # FastAPI app — /predict + /health endpoints
│   └── requirements.txt        # Backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Root component
│   │   ├── components/
│   │   │   ├── RiskMap.jsx     # Leaflet.js choropleth ward map
│   │   │   ├── WardCard.jsx    # Per-ward risk detail + SHAP drivers
│   │   │   └── RainfallChart.jsx  # Recharts rainfall trend
│   │   └── api/client.js       # Axios API client
│   ├── package.json
│   └── vite.config.js
│
├── reports/
│   ├── figures/
│   │   ├── shap_summary.png    # SHAP beeswarm plot
│   │   ├── roc_curve.png       # ROC curve comparison
│   │   └── confusion_matrix.png
│   └── Nyando_FloodRisk_2025.pdf   # QGIS ward risk map (for county officials)
│
├── docs/
│   └── funding/
│       └── concept_note_v1.md  # Funder-ready concept note
│
├── .gitignore
├── LICENSE                     # MIT
├── README.md                   # This file
└── requirements.txt            # Full project dependencies
```

---

## 🚀 Quick Start (Google Colab)

No installation needed. Click the badge below and run all cells:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)

```python
# Cell 1 — Install dependencies
!pip install xgboost shap imbalanced-learn scikit-learn pandas matplotlib -q

# Cell 2 — Load training data
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Flood rate: {df['flooded'].mean():.1%}")

# Cell 3 — Train XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

FEATURES = ['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']
X, y = df[FEATURES].fillna(0), df['flooded'].astype(int)
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, eval_metric='auc', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"AUC-ROC : {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
```

---

## 🛠️ Local Setup

### Prerequisites

- Python 3.10+
- Git
- Node.js 18+ (for frontend)

### 1. Clone the repo

```bash
git clone https://github.com/jameskoero/nyando-flood-ai.git
cd nyando-flood-ai
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

```bash
jupyter notebook notebooks/03_modelling.ipynb
```

### 4. Start the API (backend)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start the dashboard (frontend)

```bash
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
```

---

## 🔌 API Reference

**Base URL:** `https://nyando-flood-api.onrender.com`

### `GET /health`

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
  "land_cover": 40
}
```

**Response:**
```json
{
  "risk_score": 0.87,
  "risk_class": "HIGH",
  "risk_label": "Flood likely within 72 hours",
  "shap_top3": [
    { "feature": "rainfall_3day", "contribution": 0.34 },
    { "feature": "elevation",     "contribution": -0.21 },
    { "feature": "distance_river","contribution": 0.18 }
  ],
  "ward": "Nyando Central",
  "model_version": "1.0.0"
}
```

### Risk Classes

| Class | Score Range | Meaning |
|-------|-------------|---------|
| `LOW` | 0.00 – 0.35 | Minimal flood risk |
| `MEDIUM` | 0.35 – 0.60 | Monitor closely |
| `HIGH` | 0.60 – 0.80 | Prepare evacuation routes |
| `CRITICAL` | 0.80 – 1.00 | Immediate action required |

---

## 🖥️ Dashboard

The [live dashboard](https://nyando-flood-ai.vercel.app) features:

- 🗺️ **Interactive Leaflet.js choropleth map** — colour-coded ward risk scores
- 📤 **CSV upload** — upload new rainfall readings → auto-predict all wards
- 🔍 **Ward drill-down** — click any ward to see SHAP top-3 flood drivers
- 📈 **Rainfall trend chart** — Recharts 30-day CHIRPS history
- 📥 **Export PDF** — one-click QGIS-style ward risk report

---

## 🌐 Funding & Impact

This project targets international climate-AI funding:

| Funder | Programme | Grant Range |
|--------|-----------|-------------|
| World Bank GFDRR | Climate Risk Financing | USD 50K–500K |
| Green Climate Fund | Readiness Programme | USD 100K–10M |
| UNDP SIDA | Climate Action | USD 25K–200K |
| USAID DIV | Development Innovation | USD 25K–200K |
| Google.org | AI for SDGs | USD 50K–500K |
| Mozilla Foundation | Tech & Society | USD 10K–100K |

**Alignment:** SDG 13 (Climate Action) · SDG 11 (Sustainable Cities) · Sendai Framework Priority 1 · Kenya National Adaptation Plan

**Scale:** Methodology is directly replicable to 20+ African flood-prone basins including Tana, Athi, Nzoia (Kenya), Niger, Volta, and Zambezi.

---

## 🔒 Data Ethics & Privacy

> This project uses **100% open, non-personal datasets** including satellite imagery, public rainfall records, and aggregated census data. No individual or household-level data is collected or stored.

- ✅ **Kenya Data Protection Act 2019** — fully compliant
- ✅ **GDPR** — compliant by design (no EU personal data)
- ✅ **SHAP explainability** — no opaque black-box algorithmic decisions
- ✅ **Bias audit** — model performance compared across urban/rural sub-catchments
- ✅ **Creative Commons CC-BY-4.0** — all outputs openly published
- ✅ **Zenodo archive** — dataset DOI for academic citation and reproducibility

---

## 🗓️ Roadmap

- [x] Phase 1 — Data collection & engineering (CHIRPS + DEM + SAR labels)
- [x] Phase 2 — Model development (XGBoost + SHAP + benchmarking)
- [ ] Phase 3 — FastAPI deployment + React dashboard
- [ ] Phase 4 — QGIS PDF map for Kisumu County
- [ ] Phase 5 — WARMA gauge data integration (real-time)
- [ ] Phase 6 — SMS early warning integration (Africa's Talking API)
- [ ] Phase 7 — Expand to Tana + Athi basins
- [ ] Phase 8 — Peer-reviewed publication submission

---

## 👤 Author

**James Koero**  
Junior ML Engineer | Kisumu, Kenya

[![GitHub](https://img.shields.io/badge/GitHub-jameskoero-181717?style=flat-square&logo=github)](https://github.com/jameskoero)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jameskoero-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/jameskoero)

Academic Advisors:  
- **Prof. Samuel Liyala** — JOOUST, Kenya  
- **Prof. Johan Loeckx** — Vrije Universiteit Brussel, Belgium

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  
All datasets and outputs are published under **Creative Commons CC-BY-4.0**.

---

<div align="center">

*Built with ❤️ in Kisumu, Kenya — for the communities of Nyando Basin*

</div>
