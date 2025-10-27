"""
User Pydantic schemas for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user schema with common attributes."""

    email: EmailStr
    username: str


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str


class UserLogin(BaseModel):
    """Schema for user login."""

    username: str
    password: str


class UserResponse(UserBase):
    """Schema for user response."""

    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Schema for token payload data."""

    user_id: int | None = None
