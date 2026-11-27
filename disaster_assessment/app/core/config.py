"""
Configuration settings for the AI Disaster Damage Assessment system.
Uses pydantic-settings for environment-variable overrides.
"""

from functools import lru_cache
from typing import List, Tuple
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── App metadata ──────────────────────────────────────────────────────────
    APP_NAME: str = "AI Disaster Damage Assessment"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Mock mode ─────────────────────────────────────────────────────────────
    # Set MOCK_MODE=False in .env (or environment) once real .pth weights exist.
    MOCK_MODE: bool = True

    # ── Upload limits ─────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_FORMATS: List[str] = ["jpg", "jpeg", "png", "tif", "tiff"]
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/tiff"]

    # ── Model weight paths ────────────────────────────────────────────────────
    # Place your trained .pth files at these paths to exit mock mode.
    BUILDING_MODEL_PATH: str = "models/building_detector.pth"
    DAMAGE_MODEL_PATH: str = "models/damage_classifier.pth"

    # ── Image preprocessing ───────────────────────────────────────────────────
    IMAGE_HEIGHT: int = 512
    IMAGE_WIDTH: int = 512

    # ── Inference ─────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def IMAGE_SIZE(self) -> Tuple[int, int]:
        return (self.IMAGE_HEIGHT, self.IMAGE_WIDTH)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


# Module-level convenience instance
settings = get_settings()
