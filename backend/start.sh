#!/bin/bash
set -e

# Refresh live sensor data (AirNow + PurpleAir) using runtime API keys.
# Historical Socrata data and ChromaDB vectors are pre-built into the image —
# ingest.py is intentionally skipped here to stay under the 512 MB free-tier limit.
echo "==> Refreshing live sensor data (AirNow + PurpleAir)..."
python src/dataingestion.py

echo "==> Updating merged dataset..."
python src/datamerge.py

echo "==> Starting API server..."
exec uvicorn chatbot.main:app --host 0.0.0.0 --port 8000
