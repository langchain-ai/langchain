"""
Conversation and Message database models.

This module defines models for storing chat conversations and messages.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.agent import Agent
    from app.models.user import User


class Conversation(Base):
    """
    Conversation model representing a chat session.

    Attributes:
        id: Primary key.
        title: Conversation title (auto-generated or user-set).
        user_id: Foreign key to the user who owns this conversation.
        agent_id: Foreign key to the agent used in this conversation.
        created_at: Timestamp when the conversation was created.
        updated_at: Timestamp when the conversation was last updated.
        user: Relationship to the User model.
        agent: Relationship to the Agent model.
        messages: List of messages in this conversation.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, default="New Conversation")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="conversations")
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """
    Message model representing a single message in a conversation.

    Attributes:
        id: Primary key.
        conversation_id: Foreign key to the conversation.
        role: Message role ('user' or 'assistant').
        content: Message content.
        created_at: Timestamp when the message was created.
        conversation: Relationship to the Conversation model.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
