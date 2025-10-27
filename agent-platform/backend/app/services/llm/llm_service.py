"""
LLM service for managing language model interactions.

This module provides a service layer for creating and configuring
LLM instances from different providers.
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.models.agent import Agent
from app.models.llm_config import LLMConfig


class LLMService:
    """Service for creating and managing LLM instances."""

    @staticmethod
    def create_llm(agent: Agent, llm_config: LLMConfig | None = None) -> BaseChatModel:
        """
        Create an LLM instance based on agent configuration.

        Args:
            agent: The agent configuration.
            llm_config: Optional LLM configuration with API credentials.

        Returns:
            A configured LangChain chat model instance.

        Raises:
            ValueError: If the model provider is not supported.
        """
        provider = agent.model_provider.lower()
        model_name = agent.model_name

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
        }

        # Add API key if available from llm_config
        if llm_config and llm_config.api_key:
            llm_kwargs["api_key"] = llm_config.api_key

        if llm_config and llm_config.api_base:
            llm_kwargs["base_url"] = llm_config.api_base

        # Create appropriate LLM instance
        if provider == "openai":
            return ChatOpenAI(**llm_kwargs)
        elif provider == "anthropic":
            return ChatAnthropic(**llm_kwargs)
        else:
            msg = f"Unsupported model provider: {provider}"
            raise ValueError(msg)

    @staticmethod
    def get_available_models() -> dict[str, list[str]]:
        """
        Get available models for each provider.

        Returns:
            Dictionary mapping provider names to lists of model names.
        """
        return {
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ],
        }
