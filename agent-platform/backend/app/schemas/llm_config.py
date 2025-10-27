"""
LLM Configuration Pydantic schemas for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class LLMConfigBase(BaseModel):
    """Base LLM config schema."""

    provider: str = Field(..., pattern="^[a-z_]+$")
    display_name: str
    api_base: str | None = None


class LLMConfigCreate(LLMConfigBase):
    """Schema for creating a new LLM configuration."""

    api_key: str = Field(..., min_length=1)


class LLMConfigUpdate(BaseModel):
    """Schema for updating an LLM configuration."""

    display_name: str | None = None
    api_key: str | None = Field(None, min_length=1)
    api_base: str | None = None
    is_active: bool | None = None


class LLMConfigResponse(LLMConfigBase):
    """Schema for LLM config response."""

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
