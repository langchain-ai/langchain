"""Model fallback middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel


class ModelFallbackMiddleware(AgentMiddleware):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in create_agent().

    Example:
        ```python
        from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-4o",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-4o-mini, then claude-3-5-sonnet
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """Initialize model fallback middleware.

        Args:
            first_model: First fallback model (string name or instance).
            *additional_models: Additional fallbacks in order.
        """
        super().__init__()

        # Initialize all fallback models
        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Try fallback models in sequence on errors.

        Args:
            request: Initial model request.
            handler: Callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        # Try primary model first
        last_exception: Exception
        try:
            return handler(request)
        except Exception as e:  # noqa: BLE001
            last_exception = e

        # Try fallback models
        for fallback_model in self.models:
            request.model = fallback_model
            try:
                return handler(request)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Try fallback models in sequence on errors (async version).

        Args:
            request: Initial model request.
            handler: Async callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        # Try primary model first
        last_exception: Exception
        try:
            return await handler(request)
        except Exception as e:  # noqa: BLE001
            last_exception = e

        # Try fallback models
        for fallback_model in self.models:
            request.model = fallback_model
            try:
                return await handler(request)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                continue

        raise last_exception
