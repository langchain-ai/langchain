"""
Agent execution service for running AI agents.

This module provides the core logic for executing agent conversations,
managing context, and streaming responses.
"""

from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.models.agent import Agent
from app.models.conversation import Message


class AgentExecutor:
    """
    Service for executing AI agent conversations.

    This class handles the core agent execution logic, including
    message history management and streaming responses.
    """

    def __init__(self, agent: Agent, llm: BaseChatModel):
        """
        Initialize the agent executor.

        Args:
            agent: The agent configuration.
            llm: The language model to use.
        """
        self.agent = agent
        self.llm = llm

    def _build_messages(self, conversation_messages: list[Message], user_message: str) -> list:
        """
        Build the message list for the LLM.

        Args:
            conversation_messages: List of previous messages in the conversation.
            user_message: The current user message.

        Returns:
            List of LangChain message objects.
        """
        messages = [SystemMessage(content=self.agent.system_prompt)]

        # Add conversation history
        for msg in conversation_messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        return messages

    async def execute_stream(
        self,
        user_message: str,
        conversation_messages: list[Message] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute the agent and stream the response.

        Args:
            user_message: The user's input message.
            conversation_messages: Optional list of previous messages for context.

        Yields:
            Chunks of the assistant's response as they are generated.
        """
        if conversation_messages is None:
            conversation_messages = []

        messages = self._build_messages(conversation_messages, user_message)

        # Stream the response
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def execute(
        self,
        user_message: str,
        conversation_messages: list[Message] | None = None,
    ) -> str:
        """
        Execute the agent and return the complete response.

        Args:
            user_message: The user's input message.
            conversation_messages: Optional list of previous messages for context.

        Returns:
            The complete assistant response.
        """
        if conversation_messages is None:
            conversation_messages = []

        messages = self._build_messages(conversation_messages, user_message)

        # Get complete response
        response = await self.llm.ainvoke(messages)

        return response.content
