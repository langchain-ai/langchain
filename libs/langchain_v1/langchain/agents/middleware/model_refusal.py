"""Opt-in error handling for explicit model refusals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.exceptions import ModelRefusalError

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.agents.structured_output import StructuredOutputValidationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage


def _raise_if_refused(message: BaseMessage) -> None:
    for block in message.content_blocks:
        if block["type"] == "refusal":
            raise ModelRefusalError.from_refusal(block, raw_message=message)


def _check_refusal(response: ModelResponse[ResponseT]) -> ModelResponse[ResponseT]:
    for message in response.result:
        _raise_if_refused(message)
    return response


class ModelRefusalMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Raise `ModelRefusalError` for responses containing a standard refusal block.

    Add this middleware to `create_agent` to treat a refusal as an error instead of
    a successful final answer. Model calls without this middleware continue to
    return messages, including refusal content.

    List it after `ModelFallbackMiddleware` so the fallback can catch the error.
    Streaming content may already have been emitted before the refusal is known.
    This middleware does not infer refusals from text or retry blocked requests.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Check the completed synchronous model response for a refusal."""
        try:
            response = handler(request)
        except StructuredOutputValidationError as exc:
            _raise_if_refused(exc.ai_message)
            raise
        return _check_refusal(response)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Check the completed asynchronous model response for a refusal."""
        try:
            response = await handler(request)
        except StructuredOutputValidationError as exc:
            _raise_if_refused(exc.ai_message)
            raise
        return _check_refusal(response)
