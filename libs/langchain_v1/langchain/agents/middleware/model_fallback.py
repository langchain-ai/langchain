"""Model fallback middleware for agents."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage as _AIMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage


def _normalise_ai_message(message: AIMessage) -> AIMessage:
    """Convert provider-specific content blocks to LangChain standard types.

    Converts OpenAI `function_call` content blocks to standard `tool_use` blocks.
    This ensures messages produced by a fallback model (e.g. an OpenAI model)
    are safe to pass back as history to the primary model (e.g. ChatBedrockConverse),
    which only understands LangChain-standard content block types.
    """
    if isinstance(message.content, str) or not message.content:
        return message

    normalised: list[Any] = []
    changed = False
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "function_call":
            raw_args = block.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    tool_input: dict[str, Any] = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    tool_input = {}
            elif isinstance(raw_args, dict):
                tool_input = raw_args
            else:
                tool_input = {}
            tool_use_id = block.get("id") or block.get("callId") or block.get("call_id") or ""
            normalised.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": block["name"],
                    "input": tool_input,
                }
            )
            changed = True
        else:
            normalised.append(block)

    if not changed:
        return message
    return message.model_copy(update={"content": normalised})


def _normalise_fallback_response(
    response: ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    """Normalise provider-specific content blocks in a fallback model response.

    Applied after a fallback model succeeds so that the returned message history
    only contains LangChain-standard content block types before being passed
    back to the primary model.
    """
    if isinstance(response, _AIMessage):
        return _normalise_ai_message(response)

    if isinstance(response, ExtendedModelResponse):
        normalised_result = [
            _normalise_ai_message(msg) if isinstance(msg, _AIMessage) else msg
            for msg in response.model_response.result
        ]
        return replace(
            response,
            model_response=replace(response.model_response, result=normalised_result),
        )

    # ModelResponse
    normalised_result = [
        _normalise_ai_message(msg) if isinstance(msg, _AIMessage) else msg
        for msg in response.result
    ]
    return replace(response, result=normalised_result)


class ModelFallbackMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in `create_agent`.

    After a successful fallback, any provider-specific content blocks in the
    response (e.g. OpenAI `function_call` blocks) are normalised to LangChain
    standard types before the message is added to history. This ensures the
    primary model only ever sees content block types it is designed to handle.

    Example:
        ```python
        from langchain.agents.middleware import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-4o",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-4o-mini, then claude-sonnet-4-5-20250929
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

        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Try fallback models in sequence on errors.

        Args:
            request: Initial model request.
            handler: Callback to execute the model.

        Returns:
            Response from successful model call, normalised to LangChain standard
            content block types when a fallback model was used.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        last_exception: Exception
        try:
            return handler(request)
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            try:
                response = handler(request.override(model=fallback_model))
                return _normalise_fallback_response(response)
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Try fallback models in sequence on errors (async version).

        Args:
            request: Initial model request.
            handler: Async callback to execute the model.

        Returns:
            Response from successful model call, normalised to LangChain standard
            content block types when a fallback model was used.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        last_exception: Exception
        try:
            return await handler(request)
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            try:
                response = await handler(request.override(model=fallback_model))
                return _normalise_fallback_response(response)
            except Exception as e:
                last_exception = e
                continue

        raise last_exception
