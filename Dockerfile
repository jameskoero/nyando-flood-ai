FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
RUN echo "=== /app contents ===" && ls -lh /app
RUN echo "=== /app/models ===" && ls -lh /app/models/ 2>/dev/null || echo "models/ missing"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
