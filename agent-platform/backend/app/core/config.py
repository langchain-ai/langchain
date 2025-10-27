"""
Core configuration settings for the AI Agent Platform.

This module defines all configuration settings using Pydantic Settings,
allowing configuration through environment variables.
"""

from typing import Optional

from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        PROJECT_NAME: The name of the project.
        VERSION: The current version of the application.
        API_V1_STR: API version 1 prefix.
        SECRET_KEY: Secret key for JWT token generation.
        ACCESS_TOKEN_EXPIRE_MINUTES: JWT token expiration time in minutes.
        DATABASE_URL: PostgreSQL database connection URL.
        CHROMA_PERSIST_DIR: Directory for ChromaDB persistence.
        CORS_ORIGINS: List of allowed CORS origins.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Project Info
    PROJECT_NAME: str = "AI Agent Platform"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database
    DATABASE_URL: str = "sqlite:///./agent_platform.db"

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./chroma_data"

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v


settings = Settings()
