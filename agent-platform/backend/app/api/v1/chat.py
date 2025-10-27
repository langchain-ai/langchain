"""
Chat API endpoints for agent conversations.

This module provides endpoints for creating conversations and
sending messages to agents with streaming support.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_active_user, get_db
from app.models.agent import Agent
from app.models.conversation import Conversation, Message
from app.models.llm_config import LLMConfig
from app.models.user import User
from app.schemas.conversation import (
    ChatRequest,
    ConversationCreate,
    ConversationResponse,
)
from app.services.agent.agent_executor import AgentExecutor
from app.services.llm.llm_service import LLMService

router = APIRouter()


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    conversation_data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Conversation:
    """
    Create a new conversation with an agent.

    Args:
        conversation_data: Conversation creation data.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        The created Conversation object.

    Raises:
        HTTPException: If agent not found.
    """
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.id == conversation_data.agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Create conversation
    db_conversation = Conversation(
        title=conversation_data.title or "New Conversation",
        user_id=current_user.id,
        agent_id=conversation_data.agent_id,
    )
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)

    return db_conversation


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> list[Conversation]:
    """
    List all conversations for the current user.

    Args:
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        List of Conversation objects.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return conversations


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Conversation:
    """
    Get a specific conversation with its messages.

    Args:
        conversation_id: The conversation ID.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        The Conversation object with messages.

    Raises:
        HTTPException: If conversation not found or user doesn't have access.
    """
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if conversation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )

    return conversation


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: int,
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Send a message to an agent and stream the response.

    Args:
        conversation_id: The conversation ID.
        chat_request: The chat request containing the user message.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        Streaming response with the agent's reply.

    Raises:
        HTTPException: If conversation not found or user doesn't have access.
    """
    # Get conversation
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if conversation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )

    # Get agent
    agent = db.query(Agent).filter(Agent.id == conversation.agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Save user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=chat_request.message,
    )
    db.add(user_message)
    db.commit()

    # Get LLM configuration
    llm_config = (
        db.query(LLMConfig)
        .filter(
            LLMConfig.provider == agent.model_provider,
            LLMConfig.is_active == True,
        )
        .first()
    )

    # Create LLM and executor
    llm = LLMService.create_llm(agent, llm_config)
    executor = AgentExecutor(agent, llm)

    # Get conversation history
    previous_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    # Stream response
    async def response_generator():
        """Generate streaming response and save to database."""
        full_response = ""

        async for chunk in executor.execute_stream(
            chat_request.message,
            previous_messages[:-1],  # Exclude the just-added user message
        ):
            full_response += chunk
            yield chunk

        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response,
        )
        db.add(assistant_message)
        db.commit()

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a conversation.

    Args:
        conversation_id: The conversation ID.
        current_user: The authenticated user.
        db: Database session.

    Raises:
        HTTPException: If conversation not found or user doesn't have access.
    """
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if conversation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this conversation",
        )

    db.delete(conversation)
    db.commit()
