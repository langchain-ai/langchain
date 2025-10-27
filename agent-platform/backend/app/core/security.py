"""
Security utilities for authentication and authorization.

This module provides functions for password hashing, JWT token generation,
and user authentication.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(subject: str | Any, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        subject: The subject (typically user ID) to encode in the token.
        expires_delta: Optional expiration time delta. If not provided,
                      uses default from settings.

    Returns:
        Encoded JWT token as a string.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: The plain text password to verify.
        hashed_password: The hashed password to compare against.

    Returns:
        `True` if the password matches, `False` otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: The plain text password to hash.

    Returns:
        The hashed password string.
    """
    return pwd_context.hash(password)
