#!/bin/bash
set -e

echo "==> Refreshing live sensor data (AirNow + PurpleAir)..."
python src/dataingestion.py

echo "==> Updating merged dataset..."
python src/datamerge.py

# Build ChromaDB index using Gemini text-embedding-004.
# No PyTorch or local model — runs entirely via the Gemini API.
echo "==> Building vector index with Gemini embeddings..."
python src/ingest.py

echo "==> Starting API server..."
exec uvicorn chatbot.main:app --host 0.0.0.0 --port 8000
