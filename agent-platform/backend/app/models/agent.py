"""
Agent database model.

This module defines the Agent model for storing AI agent configurations.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.user import User


class Agent(Base):
    """
    Agent model representing an AI agent configuration.

    Attributes:
        id: Primary key.
        name: Agent's display name.
        description: Agent's description.
        system_prompt: System prompt defining agent behavior.
        model_provider: LLM provider (e.g., 'openai', 'anthropic').
        model_name: Specific model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
        temperature: Model temperature parameter (0.0-2.0).
        max_tokens: Maximum tokens in response.
        is_published: Whether the agent is published and available for use.
        owner_id: Foreign key to the user who created this agent.
        created_at: Timestamp when the agent was created.
        updated_at: Timestamp when the agent was last updated.
        owner: Relationship to the User model.
        conversations: List of conversations using this agent.
    """

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False)

    # Model configuration
    model_provider = Column(String, nullable=False)  # 'openai', 'anthropic', etc.
    model_name = Column(String, nullable=False)  # 'gpt-4o', 'claude-3-5-sonnet', etc.
    temperature = Column(Float, default=0.7, nullable=False)
    max_tokens = Column(Integer, default=2000, nullable=False)

    # Status
    is_published = Column(Boolean, default=False, nullable=False)

    # Ownership
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    owner = relationship("User", back_populates="agents")
    conversations = relationship(
        "Conversation", back_populates="agent", cascade="all, delete-orphan"
    )
