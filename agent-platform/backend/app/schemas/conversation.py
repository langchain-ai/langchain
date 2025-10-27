"""
Conversation and Message Pydantic schemas for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class MessageBase(BaseModel):
    """Base message schema."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class MessageCreate(BaseModel):
    """Schema for creating a message."""

    content: str


class MessageResponse(MessageBase):
    """Schema for message response."""

    id: int
    conversation_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationBase(BaseModel):
    """Base conversation schema."""

    title: str | None = "New Conversation"


class ConversationCreate(BaseModel):
    """Schema for creating a conversation."""

    agent_id: int
    title: str | None = "New Conversation"


class ConversationResponse(ConversationBase):
    """Schema for conversation response."""

    id: int
    user_id: int
    agent_id: int
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    """Schema for chat request."""

    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Schema for chat response."""

    message: str
    conversation_id: int
