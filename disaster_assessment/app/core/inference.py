"""
Inference pipeline for building detection and damage assessment.

Flow
────
preprocess_image()         (preprocessing.py)
    │
    ├─► run_building_detection()    – detects buildings in the BEFORE image
    │
    └─► run_damage_assessment()     – classifies damage per building using
                                      BEFORE + AFTER images
                                          │
                                          └─► format_prediction_response()
"""

import logging
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DAMAGE_CLASSES = ["no_damage", "minor_damage", "major_damage", "destroyed"]
DAMAGE_SEVERITY = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]

# Probability weights for mock damage assignment
# Reflects realistic post-disaster distributions
_MOCK_DAMAGE_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


# ─────────────────────────────────────────────────────────────────────────────
# Building Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_building_detection(
    before_tensor: torch.Tensor,
    model: Optional[nn.Module],
    mock_mode: bool,
) -> List[Dict]:
    """
    Detect buildings in the before-disaster image.

    Parameters
    ----------
    before_tensor : (1, 3, H, W) float32 tensor
    model         : loaded building-detection model, or None
    mock_mode     : if True, return simulated bounding boxes

    Returns
    -------
    List of dicts, each with keys:
        x1, y1, x2, y2   – bounding-box pixel coordinates (int)
        confidence        – detection confidence (float, 0–1)
        is_mock           – True when simulated (bool)
    """
    if mock_mode or model is None:
        return _mock_building_detection()

    # ── INTEGRATION POINT ─────────────────────────────────────────────────
    # Replace the block below with a real call to your building detector.
    #
    #   with torch.no_grad():
    #       raw_output = model(before_tensor)
    #   buildings = postprocess_detections(raw_output)   # your function
    #   return buildings
    # ─────────────────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            _ = model(before_tensor)        # placeholder call
        logger.warning(
            "Real model called but output parsing not implemented. "
            "Falling back to mock detection."
        )
        return _mock_building_detection()
    except Exception as exc:
        logger.error("Building detection failed: %s – using mock.", exc)
        return _mock_building_detection()


def _mock_building_detection() -> List[Dict]:
    """Generate realistic-looking mock building bounding boxes."""
    num_buildings = random.randint(2, 6)
    buildings = []
    img_size = 512

    for i in range(num_buildings):
        min_dim = 30
        x1 = random.randint(0, img_size - min_dim - 1)
        y1 = random.randint(0, img_size - min_dim - 1)
        w = random.randint(min_dim, min(120, img_size - x1))
        h = random.randint(min_dim, min(120, img_size - y1))
        x2 = x1 + w
        y2 = y1 + h

        buildings.append(
            {
                "building_id": i + 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": w,
                "height": h,
                "confidence": round(random.uniform(0.60, 0.95), 3),
                "is_mock": True,
            }
        )

    logger.info("Mock building detection: %d buildings generated.", num_buildings)
    return buildings


# ─────────────────────────────────────────────────────────────────────────────
# Damage Assessment
# ─────────────────────────────────────────────────────────────────────────────

def run_damage_assessment(
    before_tensor: torch.Tensor,
    after_tensor: torch.Tensor,
    buildings: List[Dict],
    model: Optional[nn.Module],
    mock_mode: bool,
) -> List[Dict]:
    """
    Classify damage level for each detected building.

    Parameters
    ----------
    before_tensor : (1, 3, H, W) float32 tensor
    after_tensor  : (1, 3, H, W) float32 tensor
    buildings     : list of building dicts from run_building_detection()
    model         : loaded damage-classification model, or None
    mock_mode     : if True, return simulated classifications

    Returns
    -------
    List of dicts with keys:
        building_id, damage_class, severity, confidence, is_mock
    """
    if mock_mode or model is None:
        return _mock_damage_assessment(buildings)

    # ── INTEGRATION POINT ─────────────────────────────────────────────────
    # Replace the block below with a real call to your damage classifier.
    #
    #   with torch.no_grad():
    #       raw_scores = model(before_tensor, after_tensor)  # (1, 4)
    #   damage_results = postprocess_scores(raw_scores, buildings)
    #   return damage_results
    # ─────────────────────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            _ = model(before_tensor, after_tensor)  # placeholder call
        logger.warning(
            "Real model called but output parsing not implemented. "
            "Falling back to mock damage assessment."
        )
        return _mock_damage_assessment(buildings)
    except Exception as exc:
        logger.error("Damage assessment failed: %s – using mock.", exc)
        return _mock_damage_assessment(buildings)


def _mock_damage_assessment(buildings: List[Dict]) -> List[Dict]:
    """Generate mock damage classifications for each building."""
    results = []
    for building in buildings:
        class_idx = random.choices(range(4), weights=_MOCK_DAMAGE_WEIGHTS, k=1)[0]
        results.append(
            {
                "building_id": building["building_id"],
                "damage_class": DAMAGE_CLASSES[class_idx],
                "severity": DAMAGE_SEVERITY[class_idx],
                "confidence": round(random.uniform(0.55, 0.98), 3),
                "is_mock": True,
            }
        )

    logger.info("Mock damage assessment completed for %d buildings.", len(buildings))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Response Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_prediction_response(
    buildings: List[Dict],
    damage_assessments: List[Dict],
    mock_mode: bool,
    processing_time_ms: float,
) -> Dict:
    """
    Merge building detections and damage assessments into the final API response.

    Returns
    -------
    dict suitable for JSON serialisation
    """
    # Build a lookup from building_id → damage info
    damage_lookup: Dict[int, Dict] = {
        d["building_id"]: d for d in damage_assessments
    }

    merged_buildings = []
    counts = {cls: 0 for cls in DAMAGE_CLASSES}

    for b in buildings:
        bid = b["building_id"]
        damage = damage_lookup.get(bid, {})
        damage_class = damage.get("damage_class", "unknown")

        if damage_class in counts:
            counts[damage_class] += 1

        merged_buildings.append(
            {
                "building_id": bid,
                "bounding_box": {
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "width": b.get("width", b["x2"] - b["x1"]),
                    "height": b.get("height", b["y2"] - b["y1"]),
                },
                "detection_confidence": b.get("confidence", 0.0),
                "damage_class": damage_class,
                "damage_severity": damage.get("severity", "Unknown"),
                "damage_confidence": damage.get("confidence", 0.0),
                "is_mock": b.get("is_mock", True),
            }
        )

    response = {
        "status": "success",
        "mock_mode": mock_mode,
        "total_buildings": len(buildings),
        "processing_time_ms": round(processing_time_ms, 2),
        "buildings": merged_buildings,
        "summary": {
            "total_buildings": len(buildings),
            "no_damage_count": counts["no_damage"],
            "minor_damage_count": counts["minor_damage"],
            "major_damage_count": counts["major_damage"],
            "destroyed_count": counts["destroyed"],
        },
    }

    if mock_mode:
        response["mock_warning"] = (
            "⚠️  MOCK MODE ACTIVE: These predictions are computer-generated "
            "simulations for demonstration purposes only. "
            "They are NOT actual AI model predictions."
        )

    return response
