"""Wrapper around Google's PaLM Chat API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from premai.models.chat_completion_response import ChatCompletionResponse
    from premai.api.chat_completions.v1_chat_completions_create import (
        ChatCompletionResponseStream,
    )

logger = logging.getLogger(__name__)


class ChatPremAPIError(Exception):
    """Error with the `PremAI` API."""


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


def _response_to_result(
    response: ChatCompletionResponse,
    stop: Optional[List[str]],
) -> ChatResult:
    """Converts a Prem API response into a LangChain result"""

    if not response.choices:
        raise ChatPremAPIError("ChatResponse must have at least one candidate")
    generations: List[ChatGeneration] = []
    for choice in response.choices:
        role = choice.message.role
        if role is None:
            raise ChatPremAPIError(f"ChatResponse {choice} must have a role.")

        # If content is None then it will be replaced by ""
        content = _truncate_at_stop_tokens(text=choice.message.content or "", stop=stop)
        if content is None:
            raise ChatPremAPIError(f"ChatResponse must have a content: {content}")

        if role == "assistant":
            generations.append(
                ChatGeneration(text=content, message=AIMessage(content=content))
            )
        elif role == "user":
            generations.append(
                ChatGeneration(text=content, message=HumanMessage(content=content))
            )
        else:
            generations.append(
                ChatGeneration(
                    text=content, message=ChatMessage(role=role, content=content)
                )
            )
        return ChatResult(generations=generations)


def _messages_to_prompt_dict(input_messages: List[BaseMessage]) -> dict:
    """Converts a list of LangChain Messages into a simple dict which is the message structure in Prem"""

    remaining = list(enumerate(input_messages))
    system_prompt: str = None
    examples_and_messages: List[Dict[str, str]] = []

    while remaining:
        index, input_message = remaining.pop(0)

        if isinstance(input_message, SystemMessage):
            if index != 0:
                raise ChatPremAPIError("System message must be first input message.")
            system_prompt = cast(str, input_message.content)

        elif isinstance(input_message, HumanMessage) and input_message.example:
            if examples_and_messages:
                raise ChatPremAPIError(
                    "Message examples must come before other messages."
                )

            _, next_input_message = remaining.pop(0)
            if isinstance(next_input_message, AIMessage) and next_input_message.example:
                examples_and_messages.extend(
                    [
                        {"role": "user", "content": input_message.content},
                        {"role": "assistant", "content": next_input_message.content},
                    ]
                )
            else:
                raise ChatPremAPIError(
                    "Human example message must be immediately followed by an "
                    " AI example response."
                )
        elif isinstance(input_message, AIMessage) and input_message.example:
            raise ChatPremAPIError(
                "AI example message must be immediately preceded by a Human "
                "example message."
            )
        elif isinstance(input_message, AIMessage):
            examples_and_messages.append(
                {"role": "assistant", "content": input_message.content}
            )
        elif isinstance(input_message, HumanMessage):
            examples_and_messages.append(
                {"role": "user", "content": input_message.content}
            )
        else:
            raise ChatPremAPIError(
                "Messages without an explicit role not supported by PremAI API."
            )
        return system_prompt, examples_and_messages


def _create_retry_decorator() -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle PremAI exceptions"""

    import premai.models

    errors = [
        premai.models.api_response_validation_error.APIResponseValidationError,
        premai.models.conflict_error.ConflictError,
        premai.models.model_not_found_error.ModelNotFoundError,
        premai.models.permission_denied_error.PermissionDeniedError,
        premai.models.provider_api_connection_error.ProviderAPIConnectionError,
        premai.models.provider_api_status_error.ProviderAPIStatusError,
        premai.models.provider_api_timeout_error.ProviderAPITimeoutError,
        premai.models.provider_internal_server_error.ProviderInternalServerError,
        premai.models.provider_not_found_error.ProviderNotFoundError,
        premai.models.rate_limit_error.RateLimitError,
        premai.models.unprocessable_entity_error.UnprocessableEntityError,
        premai.models.validation_error.ValidationError,
    ]

    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=retry_if_exception_type(*errors),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
