FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app /app/app
COPY ml /app/ml
COPY data /app/data

ENV PYTHONPATH=/app:/app/ml/src
ENV DEMO_MODE=true
ENV MODEL_PATH=/app/ml/models/best_model
ENV DEMO_DATA_PATH=/app/data/demo/thermal_detections_demo.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
