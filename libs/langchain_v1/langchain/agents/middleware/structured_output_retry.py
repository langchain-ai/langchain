"""Minimal structured output retry middleware example."""

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.structured_output import StructuredOutputError


class StructuredOutputRetryMiddleware(AgentMiddleware):
    """Retries model calls when structured output parsing fails."""

    def __init__(self, max_retries: int = 2) -> None:
        """Initialize the structured output retry middleware.

        Args:
            max_retries: Maximum number of retry attempts.
        """
        self.max_retries = max_retries

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Intercept and control model execution via handler callback."""
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except StructuredOutputError as exc:
                if attempt == self.max_retries:
                    raise

                # Include both the AI message and error in a single human message
                # to maintain valid chat history alternation
                ai_content = exc.ai_message.content
                error_message = (
                    f"Your previous response was:\n{ai_content}\n\n"
                    f"Error: {exc}. Please try again with a valid response."
                )
                request.messages.append(HumanMessage(content=error_message))

        # This should never be reached, but satisfies type checker
        return handler(request)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Intercept and control async model execution via handler callback."""
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except StructuredOutputError as exc:
                if attempt == self.max_retries:
                    raise

                # Include both the AI message and error in a single human message
                # to maintain valid chat history alternation
                ai_content = exc.ai_message.content
                error_message = (
                    f"Your previous response was:\n{ai_content}\n\n"
                    f"Error: {exc}. Please try again with a valid response."
                )
                request.messages.append(HumanMessage(content=error_message))

        # This should never be reached, but satisfies type checker
        return await handler(request)
