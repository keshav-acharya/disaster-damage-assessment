"""
Health-check endpoint.

GET /health  →  200 OK with system information
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings
from app.models.model_loader import model_registry

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    mock_mode: bool
    mock_mode_message: str
    building_detector_loaded: bool
    damage_classifier_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Returns the operational status of the backend service.

    - **status**: "ok" when the service is running.
    - **mock_mode**: True when no trained models are found; predictions will be simulated.
    """
    mock_msg = (
        "⚠️ Running in MOCK MODE – predictions are simulated. "
        "Load real .pth weights and set MOCK_MODE=False to enable real inference."
        if model_registry.is_mock_mode
        else "✅ Real models loaded. Running in production mode."
    )

    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        mock_mode=model_registry.is_mock_mode,
        mock_mode_message=mock_msg,
        building_detector_loaded=model_registry.building_detector is not None,
        damage_classifier_loaded=model_registry.damage_classifier is not None,
    )
