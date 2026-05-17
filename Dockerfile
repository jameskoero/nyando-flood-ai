FROM python:3.12-slim

WORKDIR /app

# Pin exact versions matching training environment
RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 \
    numpy==2.0.2 \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    joblib==1.3.2 \
    pandas==2.2.2 \
    shap==0.45.0

COPY . .

# Confirm model exists at build time — fails loudly if missing
RUN python -c "import joblib; joblib.load('backend/models/nyando_gb_v1.pkl'); print('Model OK')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
