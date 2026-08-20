"""Chat models for conversational AI."""

from __future__ import annotations

import asyncio
import builtins  # noqa: TC003  # runtime-evaluated; subclass `dict()` shadows the builtin
import contextlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from langchain_protocol.protocol import MessageFinishData
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core._api import beta, deprecated, suppress_langchain_deprecation_warning
from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.globals import get_llm_cache
from langchain_core.language_models._compat_bridge import (
    achunks_to_events,
    amessage_to_events,
    chunks_to_events,
    message_to_events,
)
from langchain_core.language_models._utils import (
    _filter_invocation_params_for_tracing,
    _normalize_messages,
    _update_message_content_to_blocks,
)
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.language_models.model_profile import (
    ModelProfile,
    _warn_unknown_profile_keys,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    convert_to_messages,
    is_data_content_block,
    message_chunk_to_message,
)
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_image_block,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
    RunInfo,
)
from langchain_core.outputs.chat_generation import merge_chat_generation_chunks
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableBinding, RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers._streaming import (
    _StreamingCallbackHandler,
    _V2StreamingCallbackHandler,
)
from langchain_core.utils._gateway import GATEWAY_METADATA_RESPONSE_KEY
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import LC_ID_PREFIX, from_env

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable

    from langchain_protocol.protocol import MessagesData

    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool


def _generate_response_from_error(error: BaseException) -> list[ChatGeneration]:
    if hasattr(error, "response"):
        response = error.response
        metadata: dict[str, Any] = {}
        if hasattr(response, "json"):
            try:
                metadata["body"] = response.json()
            except Exception:
                try:
                    metadata["body"] = getattr(response, "text", None)
                except Exception:
                    metadata["body"] = None
        if hasattr(response, "headers"):
            try:
                metadata["headers"] = dict(response.headers)
            except Exception:
                metadata["headers"] = None
        if hasattr(response, "status_code"):
            metadata["status_code"] = response.status_code
        if hasattr(error, "request_id"):
            metadata["request_id"] = error.request_id
        generations = [
            ChatGeneration(message=AIMessage(content="", response_metadata=metadata))
        ]
    else:
        generations = []

    return generations


def _format_for_tracing(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Format messages for tracing in `on_chat_model_start`.

    - Update image content blocks to OpenAI Chat Completions format (backward
    compatibility).
    - Add `type` key to content blocks that have a single key.

    Args:
        messages: List of messages to format.

    Returns:
        List of messages formatted for tracing.

    """
    messages_to_trace = []
    for message in messages:
        message_to_trace = message
        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                if isinstance(block, dict):
                    # Update image content blocks to OpenAI # Chat Completions format.
                    if (
                        block.get("type") == "image"
                        and is_data_content_block(block)
                        and not ("file_id" in block or block.get("source_type") == "id")
                    ):
                        if message_to_trace is message:
                            # Shallow copy
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = (  # type: ignore[index]  # mypy confused by .model_copy
                            convert_to_openai_image_block(block)
                        )
                    elif (
                        block.get("type") == "file"
                        and is_data_content_block(block)  # v0 (image/audio/file) or v1
                        and "base64" in block
                        # Backward compat: convert v1 base64 blocks to v0
                    ):
                        if message_to_trace is message:
                            # Shallow copy
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = {  # type: ignore[index]
                            **{k: v for k, v in block.items() if k != "base64"},
                            "data": block["base64"],
                            "source_type": "base64",
                        }
                    elif len(block) == 1 and "type" not in block:
                        # Tracing assumes all content blocks have a "type" key. Here
                        # we add this key if it is missing, and there's an obvious
                        # choice for the type (e.g., a single key in the block).
                        if message_to_trace is message:
                            # Shallow copy
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)
                        key = next(iter(block))
                        message_to_trace.content[idx] = {  # type: ignore[index]
                            "type": key,
                            key: block[key],
                        }
        messages_to_trace.append(message_to_trace)

    return messages_to_trace


def generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    """Generate from a stream.

    Args:
        stream: Iterator of `ChatGenerationChunk`.

    Raises:
        ValueError: If no generations are found in the stream.

    Returns:
        Chat result.

    """
    generation = next(stream, None)
    if generation:
        generation += list(stream)
    if generation is None:
        msg = "No generations found in stream."
        raise ValueError(msg)
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


async def agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    """Async generate from a stream.

    Args:
        stream: AsyncIterator of `ChatGenerationChunk`.

    Returns:
        Chat result.

    """
    chunks = [chunk async for chunk in stream]
    return await run_in_executor(None, generate_from_stream, iter(chunks))


def _format_ls_structured_output(
    ls_structured_output_format: dict[str, Any] | None,
) -> dict[str, Any]:
    if ls_structured_output_format:
        try:
            ls_structured_output_format_dict = {
                "ls_structured_output_format": {
                    "kwargs": ls_structured_output_format.get("kwargs", {}),
                    "schema": convert_to_json_schema(
                        ls_structured_output_format["schema"]
                    ),
                }
            }
        except ValueError:
            ls_structured_output_format_dict = {}
    else:
        ls_structured_output_format_dict = {}