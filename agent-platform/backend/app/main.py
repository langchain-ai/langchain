"""
Main FastAPI application entry point.

This module initializes the FastAPI application, configures CORS,
and sets up all API routes.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes the database on startup.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application.
    """
    # Startup
    init_db()
    yield
    # Shutdown (cleanup if needed)


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint.

    Returns:
        Welcome message.
    """
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy"}
