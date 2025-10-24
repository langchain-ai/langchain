"""Minimal structured output retry middleware example."""

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import StructuredOutputError


class StructuredOutputRetryMiddleware(AgentMiddleware):
    """Retries model calls when structured output parsing fails."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except StructuredOutputError as exc:
                if attempt >= self.max_retries:
                    raise

                # Add error feedback for retry
                if exc.ai_message:
                    request.messages.append(exc.ai_message)

                request.messages.append(
                    HumanMessage(
                        content=f"Error: {exc}. Please try again with a valid response."
                    )
                )

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except StructuredOutputError as exc:
                if attempt >= self.max_retries:
                    raise

                if exc.ai_message:
                    request.messages.append(exc.ai_message)

                request.messages.append(
                    HumanMessage(
                        content=f"Error: {exc}. Please try again with a valid response."
                    )
                )
