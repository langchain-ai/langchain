"""Opt-in error handling for explicit model refusals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core import exceptions
from langchain_core.messages import AIMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _check_refusal(response: ModelResponse[ResponseT]) -> ModelResponse[ResponseT]:
    for message in response.result:
        if isinstance(message, AIMessage) and any(
            block["type"] == "refusal" for block in message.content_blocks
        ):
            raise exceptions.ModelRefusalError(message)
    return response


class ModelRefusalMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Raise `ModelRefusalError` for responses containing a standard refusal block.

    Add this middleware to `create_agent` (or `create_deep_agent`) to treat a
    refusal as an error instead of a successful final answer. Model calls without
    this middleware continue to return messages, including refusal content.

    Place this after an outer fallback middleware to let it catch the error.
    Streaming content may already have been emitted before the refusal is known.
    This middleware does not infer refusals from text or retry blocked requests.
    """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Check the completed synchronous model response for a refusal."""
        return _check_refusal(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Check the completed asynchronous model response for a refusal."""
        return _check_refusal(await handler(request))
