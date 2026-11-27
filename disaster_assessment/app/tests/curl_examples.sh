#!/usr/bin/env bash
# ============================================================
# curl examples for the AI Disaster Damage Assessment API
# Run these after starting the FastAPI server:
#   uvicorn app.main:app --reload
# ============================================================

BASE="http://localhost:8000"

echo ""
echo "=== 1. Health Check ==="
curl -s "$BASE/health" | python -m json.tool

echo ""
echo "=== 2. Predict (replace paths with real image files) ==="
# curl -s -X POST "$BASE/predict" \
#   -F "before_image=@/path/to/before.jpg" \
#   -F "after_image=@/path/to/after.jpg" | python -m json.tool

echo "Uncomment the predict block above and replace image paths."
echo ""
echo "=== 3. Swagger UI ==="
echo "Open in browser: $BASE/docs"
