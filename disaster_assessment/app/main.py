"""
FastAPI application entry point.

Start the server with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.model_loader import initialize_models
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description=(
            "Backend API for AI-powered disaster damage assessment from satellite imagery.\n\n"
            "Analyses before-and-after satellite images to detect buildings and classify "
            "structural damage caused by floods, earthquakes, or hurricanes.\n\n"
            "**⚠️ Note**: The system currently runs in **MOCK MODE** "
            "(no trained model weights found). "
            "Predictions are simulated for demonstration purposes."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS – allow the Streamlit frontend to call the API ─────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(predict_router)

    # ── Startup / shutdown events ─────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        logger.info("=" * 60)
        logger.info("  %s  v%s", settings.APP_NAME, settings.VERSION)
        logger.info("=" * 60)
        initialize_models()
        logger.info("Server ready.  Docs: http://localhost:8000/docs")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Server shutting down.")

    return app


app = create_app()
