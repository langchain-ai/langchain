"""Model fallback middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Generator

    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.runtime import Runtime
    from langgraph.typing import ContextT


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
            "anthropic:claude-3-5-sonnet-20241022",  # Then this
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

    def on_model_call(
        self,
        request: ModelRequest,
        state: Any,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> Generator[ModelRequest, ModelResponse, ModelResponse]:
        """Try fallback models in sequence on errors.

        Args:
            request: Initial model request.
            state: Current agent state.
            runtime: LangGraph runtime.

        Yields:
            ModelRequest to execute.

        Receives:
            ModelResponse from each attempt.

        Returns:
            Final ModelResponse (success or last error).
        """
        # Try primary model first
        current_request = request
        response = yield current_request

        # If success, return immediately
        if response.action == "return":
            return response

        # Try each fallback model
        for fallback_model in self.models:
            current_request.model = fallback_model
            response = yield current_request

            if response.action == "return":
                return response

        # All models failed, return last error
        return response
