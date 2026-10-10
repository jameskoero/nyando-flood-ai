FROM python:3.11-slim

WORKDIR /app

# The image installs exactly what CI tests and audits (constraints-ci.txt), so production cannot drift from the tested set (rows D32, D48). The API takes JSON only: no multipart parser.
COPY constraints-ci.txt .
RUN pip install --no-cache-dir -c constraints-ci.txt \
    fastapi \
    uvicorn \
    pydantic \
    slowapi \
    onnxruntime \
    numpy \
    pandas \
    scikit-learn \
    joblib

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

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin nyando
USER nyando

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
