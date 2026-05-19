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
    python-multipart==0.0.9

COPY . .

RUN python -c "\
import joblib, os; \
paths = ['backend/models/nyando_xgb_v1.pkl', 'models/nyando_xgb_v1.pkl']; \
found = [p for p in paths if os.path.exists(p)]; \
print('Found:', found); \
m = joblib.load(found[0]) if found else None; \
print('Model OK:', type(m) if m else 'NOT FOUND')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
