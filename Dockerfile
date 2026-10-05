FROM python:3.11-slim

WORKDIR /app

# Install pre-built wheels — NO compilation
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    joblib==1.3.2 || \
    pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    joblib==1.3.2

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.1 \
    python-multipart==0.0.9 \
    slowapi==0.1.9 \
    onnxruntime==1.30.0

COPY . .

RUN python -c "\
from backend.integrity import load_verified_pickle; \
from backend.registered import load_registered; \
import joblib; \
m, _ = load_verified_pickle('backend/models/nyando_xgb_v1.pkl', '/app', joblib.load); \
r = load_registered('/app'); \
print('Legacy model verified and loaded:', type(m).__name__); \
print('Registered model:', r.describe()); \
assert r.session is not None, r.reason"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
