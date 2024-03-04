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
