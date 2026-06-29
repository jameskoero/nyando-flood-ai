# CONCEPT NOTE — AI-Powered Flood Early Warning System
## Nyando River Basin · Kisumu County, Kenya

**Version:** v1.0 · May 2026
**GitHub:** github.com/jameskoero/nyando-flood-ai

---

**Applicant:** James Onyango Koero — ML Engineer, Kisumu, Kenya
**Contact:** jmskoero@gmail.com · linkedin.com/in/jameskoero
**Advisors:** Prof. Johan Loeckx (VUB AI Lab, Belgium) · Prof. Samuel Liyala (JOOUST, Kenya)
**Funding Request:** USD 25,000 – 150,000 (scalable)

---

## 1. Problem Statement

The Nyando River Basin floods almost every April–May rainy season, affecting **161,000+ people** annually across Kisumu, Kericho, and Nandi Counties. Floods destroy crops worth **KES 500M+** per year.

Current early warnings arrive fewer than **6 hours** before flooding — insufficient for safe evacuation. No open, locally-trained AI system exists for this basin.

An independent published study (Awino & Machanda, 2024, arXiv:2512.13710) on this exact watershed confirms both the problem and the feasibility of the ML approach.

---

## 2. Solution — What We Have Built

| Component | Status | Evidence |
|---|---|---|
| GradientBoosting Model | ✅ Trained | AUC-ROC 0.9717 · F1 0.9022 · CV 0.9727 ± 0.0040 |
| FastAPI Backend | ✅ Live | nyando-flood-api.onrender.com/docs |
| Training Dataset | ✅ Open | 5,000 points · 6 satellite features · CC-BY-4.0 |
| GitHub Repository | ✅ Public | github.com/jameskoero/nyando-flood-ai |
| CI Pipeline | ✅ Green | 15 automated tests pass on every commit |
| React Dashboard | 🔄 Dev | Target Q3 2026 — Leaflet.js choropleth map |
| SMS Alerts | 📋 Planned | Africa's Talking API · 10,000 households |

---

## 3. Model Performance (Real GEE Satellite Data)

| Metric | Score | Interpretation |
|---|---|---|
| AUC-ROC | 0.9717 | Near-perfect flood/no-flood discrimination |
| F1-Score | 0.9022 | High balance — minimises false alarms and missed floods |
| Precision | 0.8830 | 88.3% of HIGH/CRITICAL alerts are genuine flood events |
| Recall | 0.9222 | 92.2% of real flood zones correctly identified |
| Brier Score | 0.0736 | Well-calibrated probability estimates |
| CV AUC (5-fold) | 0.9727 ± 0.0040 | Stable — generalises well across spatial folds |

All results from **real Google Earth Engine satellite data** (2,308 observation points, Nyando Basin, Kenya).

---

## 4. Data Sources — 100% Open, Non-Personal

| Feature | Source | License |
|---|---|---|
| Elevation, Slope | NASA NASADEM (GEE) | Public Domain |
| 3-Day Rainfall | CHIRPS v2 (UCSB) | CC-BY-4.0 |
| Flood Labels | Sentinel-1 SAR (ESA Copernicus) | Free Copernicus |
| Soil Clay 0-5cm | ISRIC SoilGrids | CC-BY-4.0 |
| River Distance | HydroSHEDS / OpenStreetMap | ODbL |
| Land Cover | ESA WorldCover 2021 | CC-BY-4.0 |

**Privacy & Compliance:** All data is satellite-derived. Zero personal or household data.
Fully compliant with **Kenya Data Protection Act 2019** and **GDPR**.

---

## 5. Impact Metrics

| Metric | v1.0 (Current) | v2.0 (With Funding) |
|---|---|---|
| Lead time | 72 hours | 120 hours (WARMA gauges) |
| Wards covered | 42 (Nyando sub-county) | All Kisumu County (7 sub-counties) |
| Population protected | 161,000+ | 500,000+ |
| Alert channels | API + Dashboard | + SMS 10,000 households |
| Basins | Nyando | + Tana + Nzoia (Kenya) |

---

## 6. Budget

### Option A — Core Completion (USD 25,000 · 12 months)

| Item | USD |
|---|---|
| Developer time (12 months) | 12,000 |
| React dashboard + SMS | 4,000 |
| WARMA gauge API access | 2,000 |
| Cloud hosting (12 months) | 1,500 |
| Field validation workshops ×3 | 2,500 |
| Publication + conference | 1,500 |
| Contingency (10%) | 1,500 |
| **Total** | **25,000** |

---

## 7. Team

**James Onyango Koero** — Lead Developer.
B.Sc. Physics & Mathematics, Moi University (2012).
Self-taught ML engineer with 5 deployed projects: Nyando Flood AI (AUC 0.9717), AfriSalaries (8-country salary platform), Loan Risk (Gini 0.74, Basel III), Titanic Analysis (AUC 0.85+).
Kisumu, Kenya.

**Prof. Johan Loeckx** — Academic Advisor.
VUB AI Lab, Belgium. 40+ years AI research. Peer-review and academic supervision.

**Prof. Samuel Liyala** — Local Academic Advisor.
JOOUST, Kenya. Field access and local community engagement.

---

## 8. SDG Alignment

| Framework | Alignment |
|---|---|
| SDG 13 — Climate Action | Direct: AI flood adaptation |
| SDG 11 — Sustainable Cities | Ward-level resilience planning |
| Sendai Framework Priority 1 | Understanding disaster risk through data |
| Kenya National Adaptation Plan | Flood priority action 3.2 |

---

## 9. Reproducibility & Open Science

- Full training pipeline in 4 Jupyter notebooks (GEE → EDA → Modelling → Analysis)
- All code MIT-licensed at github.com/jameskoero/nyando-flood-ai
- Training data CC-BY-4.0 (dataset DOI via Zenodo planned)
- 15 automated CI tests on every commit
- Live API endpoint for independent verification: nyando-flood-api.onrender.com/docs

---

> *"The Nyando Basin floods every year. The communities know it is coming. They just do not know when. This project gives them 72 hours to prepare."*
>
> — James Koero · Kisumu, Kenya

**github.com/jameskoero/nyando-flood-ai | jmskoero@gmail.com | linkedin.com/in/jameskoero**
