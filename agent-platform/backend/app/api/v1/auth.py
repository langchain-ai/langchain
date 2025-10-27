"""
Authentication API endpoints.

This module provides endpoints for user registration and login.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import User
from app.schemas.user import Token, UserCreate, UserLogin, UserResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)) -> User:
    """
    Register a new user account.

    Args:
        user_data: User registration data including email, username, and password.
        db: Database session.

    Returns:
        The created user object.

    Raises:
        HTTPException: If email or username already exists.
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Check if username already exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=Token)
def login(user_data: UserLogin, db: Session = Depends(get_db)) -> dict[str, str]:
    """
    Login and obtain an access token.

    Args:
        user_data: User login credentials (username and password).
        db: Database session.

    Returns:
        Dictionary containing access token and token type.

    Raises:
        HTTPException: If credentials are invalid.
    """
    # Find user by username
    user = db.query(User).filter(User.username == user_data.username).first()

    # Validate password
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account",
        )

    # Create access token
    access_token = create_access_token(subject=user.id)

    return {"access_token": access_token, "token_type": "bearer"}
