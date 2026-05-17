FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo "=== /app/backend/models ===" && ls -lh /app/backend/models/ 2>/dev/null || echo "models/ missing"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
