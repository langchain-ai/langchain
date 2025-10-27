"""
User database model.

This module defines the User model for storing user accounts.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.agent import Agent
    from app.models.conversation import Conversation


class User(Base):
    """
    User model representing a platform user.

    Attributes:
        id: Primary key.
        email: User's email address (unique).
        username: User's username (unique).
        hashed_password: Bcrypt hashed password.
        is_active: Whether the user account is active.
        is_superuser: Whether the user has admin privileges.
        created_at: Timestamp when the user was created.
        updated_at: Timestamp when the user was last updated.
        agents: List of agents created by this user.
        conversations: List of conversations initiated by this user.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    agents = relationship("Agent", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
