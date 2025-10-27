"""
Dependency injection functions for FastAPI.

This module provides reusable dependencies for authentication,
database access, and authorization.
"""

from typing import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.user import User

security = HTTPBearer()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.

    Yields:
        Database session that will be closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that extracts and validates the current user from JWT token.

    Args:
        credentials: HTTP Bearer token credentials.
        db: Database session.

    Returns:
        The authenticated User object.

    Raises:
        HTTPException: If token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency that ensures the current user is active.

    Args:
        current_user: The current authenticated user.

    Returns:
        The active User object.

    Raises:
        HTTPException: If user is inactive.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
