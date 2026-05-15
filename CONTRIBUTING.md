# Contributing to Nyando Flood AI

## Setup
```bash
git clone https://github.com/jameskoero/nyando-flood-ai.git
cd nyando-flood-ai && pip install -r requirements.txt
pytest tests/ -v   # 15 tests should pass
```

## Real Data
The training data uses real GEE satellite features (2,308 observation points).
To refresh with new satellite data: run `notebooks/01_gee_data_extraction.ipynb` in Colab.

## Priority Contributions
1. Extend GEE flood label extraction (improve SAR threshold)
2. Add UNOSAT historical flood polygons for multi-year labels
3. React dashboard (Leaflet.js choropleth)
4. SMS alerts via Africa's Talking
5. Temporal validation (train 2014-2022, test 2023-2024)

## Commit Style
```
feat: add Tana Basin real GEE data
fix: improve SAR flood detection threshold
data: add UNOSAT 2020 flood polygon labels
```
