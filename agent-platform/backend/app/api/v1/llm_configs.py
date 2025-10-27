"""
LLM Configuration API endpoints.

This module provides endpoints for managing LLM provider configurations
(admin only).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_active_user, get_db
from app.models.llm_config import LLMConfig
from app.models.user import User
from app.schemas.llm_config import LLMConfigCreate, LLMConfigResponse, LLMConfigUpdate

router = APIRouter()


def verify_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Verify that the current user is an admin.

    Args:
        current_user: The authenticated user.

    Returns:
        The admin user.

    Raises:
        HTTPException: If user is not an admin.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


@router.get("/", response_model=list[LLMConfigResponse])
def list_llm_configs(
    current_user: User = Depends(verify_admin),
    db: Session = Depends(get_db),
) -> list[LLMConfig]:
    """
    List all LLM configurations (admin only).

    Args:
        current_user: The authenticated admin user.
        db: Database session.

    Returns:
        List of LLMConfig objects.
    """
    configs = db.query(LLMConfig).all()
    return configs


@router.post("/", response_model=LLMConfigResponse, status_code=status.HTTP_201_CREATED)
def create_llm_config(
    config_data: LLMConfigCreate,
    current_user: User = Depends(verify_admin),
    db: Session = Depends(get_db),
) -> LLMConfig:
    """
    Create a new LLM configuration (admin only).

    Args:
        config_data: LLM configuration data.
        current_user: The authenticated admin user.
        db: Database session.

    Returns:
        The created LLMConfig object.

    Raises:
        HTTPException: If provider already exists.
    """
    # Check if provider already exists
    existing = db.query(LLMConfig).filter(LLMConfig.provider == config_data.provider).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Configuration for this provider already exists",
        )

    db_config = LLMConfig(**config_data.model_dump())
    db.add(db_config)
    db.commit()
    db.refresh(db_config)

    return db_config


@router.put("/{config_id}", response_model=LLMConfigResponse)
def update_llm_config(
    config_id: int,
    config_data: LLMConfigUpdate,
    current_user: User = Depends(verify_admin),
    db: Session = Depends(get_db),
) -> LLMConfig:
    """
    Update an LLM configuration (admin only).

    Args:
        config_id: The configuration ID.
        config_data: Updated configuration data.
        current_user: The authenticated admin user.
        db: Database session.

    Returns:
        The updated LLMConfig object.

    Raises:
        HTTPException: If configuration not found.
    """
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM configuration not found",
        )

    update_data = config_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)

    db.commit()
    db.refresh(config)

    return config


@router.delete("/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_llm_config(
    config_id: int,
    current_user: User = Depends(verify_admin),
    db: Session = Depends(get_db),
) -> None:
    """
    Delete an LLM configuration (admin only).

    Args:
        config_id: The configuration ID.
        current_user: The authenticated admin user.
        db: Database session.

    Raises:
        HTTPException: If configuration not found.
    """
    config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="LLM configuration not found",
        )

    db.delete(config)
    db.commit()
