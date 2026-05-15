# Model Card — Nyando Flood Risk AI v1.0
## Developed by James Koero · Kisumu, Kenya · May 2026

---

## Model Details

| Field | Value |
|---|---|
| **Model Type** | GradientBoostingClassifier (scikit-learn 1.4+) |
| **Version** | v1.0.0 |
| **File** | `models/nyando_xgb_v1.pkl` |
| **Input** | 6 real satellite features (see below) |
| **Output** | Flood probability [0.0–1.0] + risk class |
| **Serialisation** | joblib |
| **Python** | 3.10+ |

## Hyperparameters

```
n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.80, random_state=42
```

## Training Data

- **Source:** Real Google Earth Engine satellite data
- **Points:** 2,308 real GEE observations
- **Bounds:** lon 34.70–35.40°E, lat 0.40°S–0.10°N (Nyando sub-county, Kenya)
- **Flood labels:** Physics-calibrated; 2 Sentinel-1 SAR-confirmed flood anchors
- **Flood rate:** 22% calibrated

## Performance (Real GEE Data)

| Metric | Score |
|---|---|
| AUC-ROC | 0.9717 |
| F1-Score | 0.9022 |
| Precision | 0.8830 |
| Recall | 0.9222 |
| Brier Score | 0.0736 |
| CV AUC (5-fold) | 0.9727 ± 0.0040 |

## Feature Importance

```
elevation        ████████████████████ 0.31
rainfall_3day    ████████████████     0.26
distance_river   ████████████         0.19
slope            ████████             0.13
clay_percent     █████                0.08
land_cover       ██                   0.03
```

Physically sensible: low elevation + high rainfall + close river = flood ✅

## Risk Classification

| Class | Score | Action |
|---|---|---|
| LOW | 0.00–0.35 | No action |
| MEDIUM | 0.35–0.60 | Monitor |
| HIGH | 0.60–0.80 | Prepare evacuation |
| CRITICAL | 0.80–1.00 | Immediate action |

## Limitations

1. SAR flood labels from single event (April 2024) — only 2 confirmed pixels
2. Trained on Nyando sub-county only — retraining needed for other basins
3. CHIRPS v2 ~5km rainfall resolution — misses hyper-local variation
4. Static model — no real-time update without new CHIRPS query
5. WorldPop 2020 population data — may undercount recent urban growth

## Bias Audit

Performance evaluated across low/mid/high elevation zones.
All zones AUC > 0.85. No significant spatial bias detected.

## Ethics

- Zero personal data — all inputs satellite-derived
- Feature importances published — full transparency
- High recall (0.9222) prioritised — missing a flood is worse than a false alarm
- MIT + CC-BY-4.0 open access

## Compliance

Kenya DPA 2019 ✅ · GDPR ✅ · FAIR data ✅

## Citation

```
Koero, James Onyango (2026). Nyando Flood AI v1.0.
https://github.com/jameskoero/nyando-flood-ai
```

*Follows Mitchell et al. (2019) model card standard.*
