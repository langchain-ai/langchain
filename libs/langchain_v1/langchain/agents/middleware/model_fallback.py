"""Model fallback middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.runtime import Runtime


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

    def retry_model_request(
        self,
        error: Exception,  # noqa: ARG002
        request: ModelRequest,
        state: AgentState,  # noqa: ARG002
        runtime: Runtime,  # noqa: ARG002
        attempt: int,
    ) -> ModelRequest | None:
        """Retry with the next fallback model.

        Args:
            error: The exception that occurred during model invocation.
            request: The original model request that failed.
            state: The current agent state.
            runtime: The langgraph runtime.
            attempt: The current attempt number (1-indexed).

        Returns:
            ModelRequest with the next fallback model, or None if all models exhausted.
        """
        # attempt 1 = primary model failed, try models[0] (first fallback)
        fallback_index = attempt - 1
        # All fallback models exhausted
        if fallback_index >= len(self.models):
            return None
        # Try next fallback model
        request.model = self.models[fallback_index]
        return request
