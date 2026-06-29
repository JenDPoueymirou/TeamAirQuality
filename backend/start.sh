#!/bin/bash
set -e

echo "==> Fetching raw data from sources..."
python src/dataingestion.py

echo "==> Merging datasets..."
python src/datamerge.py

echo "==> Embedding rows into ChromaDB..."
python src/ingest.py

echo "==> Starting API server..."
exec uvicorn chatbot.main:app --host 0.0.0.0 --port 8000
