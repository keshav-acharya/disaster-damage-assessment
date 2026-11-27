"""
Prediction endpoint.

POST /predict
  - Accepts two images: before_image and after_image (multipart/form-data).
  - Validates format, size, and dimension compatibility.
  - Runs the building-detection → damage-assessment pipeline.
  - Returns structured JSON with detections, damage classes, and confidence scores.
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.inference import (
    format_prediction_response,
    run_building_detection,
    run_damage_assessment,
)
from app.core.preprocessing import (
    prepare_model_input,
    preprocess_image,
    validate_image,
    validate_image_pair,
)
from app.models.model_loader import model_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=Dict[str, Any])
async def predict(
    before_image: UploadFile = File(..., description="Satellite image BEFORE the disaster"),
    after_image: UploadFile = File(..., description="Satellite image AFTER the disaster"),
) -> Dict[str, Any]:
    """
    Run the full disaster damage assessment pipeline.

    **Request** (multipart/form-data):
    - `before_image`: Image captured before the disaster event.
    - `after_image` : Image captured after the disaster event.

    **Response** (JSON):
    - Total buildings detected, their bounding boxes, damage class, severity, and confidence scores.
    - `mock_mode` and `mock_warning` flags clearly indicate simulated predictions.

    **Supported formats**: JPEG, PNG, TIFF  
    **Maximum file size**: 10 MB per image
    """
    start_time = time.monotonic()

    # ── 1. Read uploaded bytes ────────────────────────────────────────────────
    try:
        before_bytes = await before_image.read()
        after_bytes = await after_image.read()
    except Exception as exc:
        logger.error("Failed to read uploaded files: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded files. Please try again.",
        )

    # ── 2. Validate individual images ─────────────────────────────────────────
    before_valid, before_msg = validate_image(before_bytes, before_image.filename or "before.jpg")
    if not before_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Before image invalid: {before_msg}",
        )

    after_valid, after_msg = validate_image(after_bytes, after_image.filename or "after.jpg")
    if not after_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"After image invalid: {after_msg}",
        )

    # ── 3. Preprocess images ──────────────────────────────────────────────────
    try:
        before_array = preprocess_image(before_bytes, settings.IMAGE_SIZE)
        after_array = preprocess_image(after_bytes, settings.IMAGE_SIZE)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Preprocessing failed: {exc}",
        )

    # ── 4. Validate image pair compatibility ──────────────────────────────────
    pair_valid, pair_msg = validate_image_pair(before_array, after_array)
    if not pair_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image pair mismatch: {pair_msg}",
        )

    # ── 5. Convert to model tensors ───────────────────────────────────────────
    before_tensor = prepare_model_input(before_array)
    after_tensor = prepare_model_input(after_array)

    # ── 6. Run inference pipeline ─────────────────────────────────────────────
    mock_mode = model_registry.is_mock_mode

    buildings = run_building_detection(
        before_tensor=before_tensor,
        model=model_registry.building_detector,
        mock_mode=mock_mode,
    )

    damage_assessments = run_damage_assessment(
        before_tensor=before_tensor,
        after_tensor=after_tensor,
        buildings=buildings,
        model=model_registry.damage_classifier,
        mock_mode=mock_mode,
    )

    # ── 7. Format and return response ─────────────────────────────────────────
    elapsed_ms = (time.monotonic() - start_time) * 1000.0
    response = format_prediction_response(
        buildings=buildings,
        damage_assessments=damage_assessments,
        mock_mode=mock_mode,
        processing_time_ms=elapsed_ms,
    )

    logger.info(
        "Prediction complete: %d buildings, mock=%s, time=%.1f ms",
        response["total_buildings"],
        mock_mode,
        elapsed_ms,
    )
    return response
