FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    joblib==1.3.2 \
    pandas==2.2.2 \
    pydantic==2.7.1 \
    python-multipart==0.0.9

COPY . .

RUN python -c "import joblib; m=joblib.load('backend/models/nyando_xgb_v1.pkl'); print('Model OK:', type(m))"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
