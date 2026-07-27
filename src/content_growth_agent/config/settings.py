"""Application configuration using Pydantic BaseSettings.

Reads environment variables from `.env` and provides a typed Settings instance.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import AnyUrl


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class MemoryType(str, Enum):
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    POSTGRES = "postgres"


class Settings(BaseSettings):
    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4"

    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Memory
    memory_type: MemoryType = MemoryType.IN_MEMORY
    redis_url: Optional[AnyUrl] = None
    postgres_url: Optional[str] = None

    # App
    log_level: str = "INFO"
    debug: bool = False

    # Streamlit
    streamlit_port: int = 8501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
