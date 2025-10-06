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
    """Middleware that provides automatic model fallback on errors.

    This middleware attempts to retry failed model calls with alternative models
    in sequence. When a model call fails, it tries the next model in the fallback
    list until either a call succeeds or all models have been exhausted.

    Example:
        ```python
        from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
        from langchain.agents import create_agent

        # Create middleware with fallback models (not including primary)
        fallback = ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # First fallback
            "anthropic:claude-3-5-sonnet-20241022",  # Second fallback
        )

        agent = create_agent(
            model="openai:gpt-4o",  # Primary model
            middleware=[fallback],
        )

        # If gpt-4o fails, automatically tries gpt-4o-mini, then claude
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """Initialize the model fallback middleware.

        Args:
            first_model: The first fallback model to try when the primary model fails.
                Can be a model name string or BaseChatModel instance.
            *additional_models: Additional fallback models to try, in order.
                Can be model name strings or BaseChatModel instances.
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
            request: The initial model request.
            state: The current agent state.
            runtime: The langgraph runtime.

        Yields:
            ModelRequest: The request to execute.

        Receives (via .send()):
            ModelResponse: The response from the model call.

        Returns:
            ModelResponse: The final response to use.
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
