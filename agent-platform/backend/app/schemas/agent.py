"""
Agent Pydantic schemas for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class AgentBase(BaseModel):
    """Base agent schema with common attributes."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    system_prompt: str = Field(..., min_length=1)
    model_provider: str = Field(..., pattern="^(openai|anthropic|custom)$")
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=32000)


class AgentCreate(AgentBase):
    """Schema for creating a new agent."""

    pass


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    system_prompt: str | None = Field(None, min_length=1)
    model_provider: str | None = Field(None, pattern="^(openai|anthropic|custom)$")
    model_name: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=32000)
    is_published: bool | None = None


class AgentResponse(AgentBase):
    """Schema for agent response."""

    id: int
    is_published: bool
    owner_id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
