"""
LLM Configuration database model.

This module defines the LLMConfig model for storing LLM provider credentials.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from app.core.database import Base


class LLMConfig(Base):
    """
    LLM Configuration model for storing model provider credentials.

    Attributes:
        id: Primary key.
        provider: Provider name ('openai', 'anthropic', etc.).
        display_name: Human-readable provider name.
        api_key: Encrypted API key.
        api_base: Optional custom API base URL.
        is_active: Whether this configuration is active.
        created_at: Timestamp when the configuration was created.
        updated_at: Timestamp when the configuration was last updated.
    """

    __tablename__ = "llm_configs"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    api_key = Column(Text, nullable=False)  # Should be encrypted in production
    api_base = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
