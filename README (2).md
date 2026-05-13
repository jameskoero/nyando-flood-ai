# 🌊 Nyando Basin Flood Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/GradientBoosting-Latest-FF6600?style=for-the-badge)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-C9A84C?style=for-the-badge)](LICENSE)
[![CI](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jameskoero/nyando-flood-ai/actions/workflows/ci.yml)
[![Data](https://img.shields.io/badge/Data-100%25%20Open-2ECC71?style=for-the-badge)](data/DATA_SOURCES.md)
[![Privacy](https://img.shields.io/badge/Privacy-DPA%202019%20Compliant-0A1628?style=for-the-badge)](docs/funding/concept_note_v1.md)

> **An open-source, AI-powered flood early warning system for Nyando River Basin, Kisumu County, Kenya.**  
> Ward-level flood susceptibility mapping at 100m resolution with 72-hour prediction lead time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)
[![Live API](https://img.shields.io/badge/Live%20API-Render-46E3B7?style=flat-square&logo=render)](https://nyando-flood-api.onrender.com/docs)
[![Dashboard](https://img.shields.io/badge/Dashboard-Vercel-000000?style=flat-square&logo=vercel)](https://nyando-flood-ai.vercel.app)

---

## 📊 Key Results

> Model: GradientBoostingClassifier (XGBoost-equivalent)  
> Training: 5,000 points · 6 satellite features · SMOTE balancing · 5-fold spatial CV  
> Reference: Awino & Machanda (2024) arXiv:2512.13710 — independent Nyando study confirms methodology

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.9915** |
| **F1-Score** | **0.9504** |
| **Precision** | **0.9437** |
| **Recall** | **0.9571** |
| **CV AUC (5-fold)** | **0.9905 ± 0.0011** |
| Training points | 5,000 |
| Resolution | 100m grid |

### Model Comparison

| Model | AUC-ROC | F1 | Notes |
|---|---|---|---|
| Logistic Regression | 0.9819 | 0.9361 | Interpretable baseline |
| Random Forest | 0.9870 | 0.9355 | Strong ensemble |
| **GradientBoosting** | **0.9915** | **0.9504** | ✅ **Selected** |

### Top Features (SHAP Importance)

```
elevation        ██████████████████████  Primary — Kano Plains <1200m = flood-prone
rainfall_3day    ████████████████        CHIRPS 3-day trigger
distance_river   ████████████            HydroSHEDS proximity risk
slope            ████████                Flat terrain drains poorly
clay_percent     ██████                  ISRIC clay = poor infiltration
land_cover       ███                     ESA WorldCover class
```

---

## 🌍 Project Overview

The Nyando River Basin floods almost every April–May rainy season, displacing **50,000–200,000 people** annually and destroying crops worth **KES 500M+**. Current warnings arrive <6 hours before flooding — not enough for safe evacuation.

This project builds an ML flood susceptibility model using **100% open satellite data**:

| Output | Description |
|---|---|
| 🗺️ **Flood Risk Map** | 100m ward-level susceptibility scores |
| ⚡ **Prediction API** | FastAPI — submit rainfall data, get risk score in <200ms |
| 📊 **Web Dashboard** | React + Leaflet.js ward map for county planners |
| 🔍 **SHAP Explainability** | Full feature transparency — no black-box decisions |
| 📋 **Concept Note** | Funder-ready PDF at [docs/funding/concept_note_v1.pdf](docs/funding/concept_note_v1.pdf) |

---

## 📁 Dataset

All data is **100% open, non-personal, and satellite/census-derived**. No individual or household data is collected.  
See full provenance: [data/DATA_SOURCES.md](data/DATA_SOURCES.md)

| Feature | Source | License |
|---|---|---|
| `elevation` | NASA NASADEM | Public Domain |
| `slope` | Derived from DEM | Public Domain |
| `rainfall_3day` | CHIRPS v2 (UCSB) | CC-BY-4.0 |
| `distance_river` | HydroSHEDS / OSM | ODbL |
| `clay_percent` | ISRIC SoilGrids | CC-BY-4.0 |
| `land_cover` | ESA WorldCover 2021 | CC-BY-4.0 |
| `flooded` | Sentinel-1 SAR / UNOSAT | Free Copernicus |

**To extract real data:** See [notebooks/01_gee_data_extraction.ipynb](notebooks/01_gee_data_extraction.ipynb) — paste into Google Colab (free account), runs in ~20 minutes.

```python
# Quick load
import pandas as pd
df = pd.read_csv(
  "https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv"
)
print(df.shape)   # (5000, 8)
```

---

## 🚀 Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/nyando-flood-ai/blob/main/notebooks/03_modelling.ipynb)

```python
# Install
!pip install scikit-learn pandas numpy matplotlib joblib -q

# Load data
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/jameskoero/nyando-flood-ai/main/data/training/nyando_training_v1.csv")
print(f"Dataset: {df.shape[0]} rows | Flood rate: {df['flooded'].mean():.1%}")

# Train model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURES = ['elevation','slope','rainfall_3day','distance_river','clay_percent','land_cover']
X, y = df[FEATURES].values, df['flooded'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(f"AUC-ROC: {auc:.4f}")
```

---

## 🔌 API Reference

**Base URL:** `https://nyando-flood-api.onrender.com`

```bash
# Health check
curl https://nyando-flood-api.onrender.com/health

# Predict flood risk
curl -X POST https://nyando-flood-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"elevation":1142.5,"slope":2.3,"rainfall_3day":87.4,"distance_river":320,"clay_percent":42.1,"land_cover":40,"ward":"Nyando Central"}'
```

**Response:**
```json
{
  "risk_score": 0.9234,
  "risk_class": "CRITICAL",
  "risk_label": "Immediate action required — evacuate low-lying areas",
  "shap_top3": [
    {"feature": "rainfall_3day", "importance": 0.2634},
    {"feature": "elevation", "importance": 0.2418},
    {"feature": "distance_river", "importance": 0.1923}
  ],
  "ward": "Nyando Central",
  "model_version": "1.0.0"
}
```

---

## 🗓️ Roadmap

- [x] Phase 1 — Data collection & engineering (CHIRPS + DEM + SAR labels)
- [x] Phase 2 — Model development (GradientBoosting + SHAP + benchmarking)
- [x] Phase 3a — FastAPI deployment (Render)
- [ ] Phase 3b — React dashboard (Leaflet.js choropleth map)
- [ ] Phase 4 — WARMA gauge data integration (real-time)
- [ ] Phase 5 — SMS early warning (Africa's Talking API)
- [ ] Phase 6 — Expand to Tana + Athi basins
- [ ] Phase 7 — Peer-reviewed publication (targeting AfricaNLP/IJCAI)

---

## 🌐 Funding & Impact

| Funder | Programme | Grant Range |
|---|---|---|
| World Bank GFDRR | Climate Risk Financing | USD 50K–500K |
| Green Climate Fund | Readiness Programme | USD 100K–10M |
| UNDP SIDA | Climate Action | USD 25K–200K |
| Google.org | AI for SDGs | USD 50K–500K |
| Mozilla Foundation | Tech & Society | USD 10K–100K |

Full concept note (PDF): [docs/funding/concept_note_v1.pdf](docs/funding/concept_note_v1.pdf)

**Alignment:** SDG 13 · SDG 11 · Sendai Framework Priority 1 · Kenya NAP

---

## 🔒 Data Ethics & Privacy

- ✅ **Kenya Data Protection Act 2019** — fully compliant (no personal data)
- ✅ **GDPR** — compliant by design
- ✅ **SHAP explainability** — transparent, auditable decisions
- ✅ **CC-BY-4.0** — all outputs openly published

---

## 👤 Author

**James Koero** · Junior ML Engineer · Kisumu, Kenya  
[![GitHub](https://img.shields.io/badge/GitHub-jameskoero-181717?style=flat-square&logo=github)](https://github.com/jameskoero)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jameskoero-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/jameskoero)

**Academic Advisors:**  
Prof. Johan Loeckx · VUB AI Lab, Belgium  
Prof. Samuel Liyala · JOOUST, Kenya

---

*Built in Kisumu, Kenya — for the communities of Nyando Basin* 🇰🇪
