"""
Model loading and registry for the AI Disaster Damage Assessment system.

HOW TO CONNECT YOUR TRAINED MODELS
───────────────────────────────────
1.  Export your trained model weights:
        torch.save(model.state_dict(), "models/building_detector.pth")
        torch.save(model.state_dict(), "models/damage_classifier.pth")

2.  In load_building_detector(), replace the placeholder with your actual
    model class and load call, e.g.:
        from your_module import BuildingDetector
        model = BuildingDetector(...)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

3.  Set MOCK_MODE=False in your .env file (or environment variable).

4.  Restart the FastAPI server.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton container for all loaded models.

    Attributes
    ----------
    building_detector  : loaded building-detection model (or None in mock mode)
    damage_classifier  : loaded damage-classification model (or None in mock mode)
    is_mock_mode       : True when at least one model is unavailable
    """

    def __init__(self):
        self.building_detector: Optional[nn.Module] = None
        self.damage_classifier: Optional[nn.Module] = None
        self.is_mock_mode: bool = True


# Module-level registry instance – populated on startup
model_registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_building_detector(model_path: str) -> Optional[nn.Module]:
    """
    Load the building-detection model from *model_path*.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  INTEGRATION POINT – Building Detection Model                       ║
    ║                                                                      ║
    ║  Replace the placeholder below with your actual model architecture.  ║
    ║                                                                      ║
    ║  Expected input  : torch.Tensor of shape (1, 3, 512, 512)           ║
    ║  Expected output : list of dicts, each containing                   ║
    ║                    { "x1", "y1", "x2", "y2", "confidence" }        ║
    ║                                                                      ║
    ║  Example:                                                            ║
    ║      from models.building_detector import UNetDetector              ║
    ║      model = UNetDetector(num_classes=1)                            ║
    ║      model.load_state_dict(                                          ║
    ║          torch.load(model_path, map_location="cpu")                 ║
    ║      )                                                               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    if not os.path.exists(model_path):
        logger.warning(
            "Building detector weights not found at '%s'. "
            "Running in MOCK MODE.",
            model_path,
        )
        return None

    try:
        # ── TODO: replace with your actual model class ─────────────────────
        model = torch.load(model_path, map_location="cpu")
        # ──────────────────────────────────────────────────────────────────
        model.eval()
        logger.info("Building detector loaded from '%s'.", model_path)
        return model
    except Exception as exc:
        logger.error("Failed to load building detector: %s", exc)
        return None


def load_damage_classifier(model_path: str) -> Optional[nn.Module]:
    """
    Load the damage-classification model from *model_path*.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  INTEGRATION POINT – Damage Assessment Model                         ║
    ║                                                                      ║
    ║  Replace the placeholder below with your actual model architecture.  ║
    ║                                                                      ║
    ║  Expected input  : two tensors (before, after), each (1, 3, 512, 512)║
    ║  Expected output : torch.Tensor of shape (1, 4) with class scores   ║
    ║                    for [no_damage, minor_damage, major_damage,       ║
    ║                          destroyed]                                  ║
    ║                                                                      ║
    ║  Example:                                                            ║
    ║      from models.damage_classifier import SiameseClassifier         ║
    ║      model = SiameseClassifier(num_classes=4)                       ║
    ║      model.load_state_dict(                                          ║
    ║          torch.load(model_path, map_location="cpu")                 ║
    ║      )                                                               ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    if not os.path.exists(model_path):
        logger.warning(
            "Damage classifier weights not found at '%s'. "
            "Running in MOCK MODE.",
            model_path,
        )
        return None

    try:
        # ── TODO: replace with your actual model class ─────────────────────
        model = torch.load(model_path, map_location="cpu")
        # ──────────────────────────────────────────────────────────────────
        model.eval()
        logger.info("Damage classifier loaded from '%s'.", model_path)
        return model
    except Exception as exc:
        logger.error("Failed to load damage classifier: %s", exc)
        return None


def initialize_models() -> ModelRegistry:
    """
    Load all models and populate the global *model_registry*.

    Called once at FastAPI startup.  Returns the populated registry.
    """
    global model_registry

    if settings.MOCK_MODE:
        logger.info(
            "MOCK_MODE is enabled in config. "
            "Skipping model loading – using simulated predictions."
        )
        model_registry.is_mock_mode = True
        return model_registry

    model_registry.building_detector = load_building_detector(
        settings.BUILDING_MODEL_PATH
    )
    model_registry.damage_classifier = load_damage_classifier(
        settings.DAMAGE_MODEL_PATH
    )

    # If either model failed to load, fall back to mock mode
    model_registry.is_mock_mode = (
        model_registry.building_detector is None
        or model_registry.damage_classifier is None
    )

    if model_registry.is_mock_mode:
        logger.warning(
            "One or more models unavailable – falling back to MOCK MODE."
        )
    else:
        logger.info("All models loaded successfully. Running in REAL mode.")

    return model_registry
