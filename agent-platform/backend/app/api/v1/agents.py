"""
Agent management API endpoints.

This module provides CRUD endpoints for managing AI agents.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_active_user, get_db
from app.models.agent import Agent
from app.models.user import User
from app.schemas.agent import AgentCreate, AgentResponse, AgentUpdate

router = APIRouter()


@router.get("/", response_model=list[AgentResponse])
def list_agents(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> list[Agent]:
    """
    List all agents owned by the current user.

    Args:
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        List of Agent objects.
    """
    agents = (
        db.query(Agent)
        .filter(Agent.owner_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return agents


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Agent:
    """
    Create a new agent.

    Args:
        agent_data: Agent configuration data.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        The created Agent object.
    """
    db_agent = Agent(
        **agent_data.model_dump(),
        owner_id=current_user.id,
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)

    return db_agent


@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Agent:
    """
    Get a specific agent by ID.

    Args:
        agent_id: The agent ID.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        The Agent object.

    Raises:
        HTTPException: If agent not found or user doesn't have access.
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    if agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this agent",
        )

    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
def update_agent(
    agent_id: int,
    agent_data: AgentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Agent:
    """
    Update an existing agent.

    Args:
        agent_id: The agent ID.
        agent_data: Updated agent configuration data.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        The updated Agent object.

    Raises:
        HTTPException: If agent not found or user doesn't have access.
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    if agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this agent",
        )

    # Update agent fields
    update_data = agent_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agent, field, value)

    db.commit()
    db.refresh(agent)

    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent(
    agent_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> None:
    """
    Delete an agent.

    Args:
        agent_id: The agent ID.
        current_user: The authenticated user.
        db: Database session.

    Raises:
        HTTPException: If agent not found or user doesn't have access.
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    if agent.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this agent",
        )

    db.delete(agent)
    db.commit()
