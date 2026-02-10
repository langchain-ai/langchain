"""Anthropic chat models."""

from __future__ import annotations

import copy
import datetime
import json
import re
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from functools import cached_property
from operator import itemgetter
from typing import Any, Final, Literal, cast

import anthropic
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import ContextOverflowError, OutputParserException
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages import content as types
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import NotRequired, Self, TypedDict

from langchain_anthropic._client_utils import (
    _get_default_async_httpx_client,
    _get_default_httpx_client,
)
from langchain_anthropic._compat import _convert_from_v1_to_anthropic
from langchain_anthropic.data._profiles import _PROFILES
from langchain_anthropic.output_parsers import extract_tool_calls

_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
}

_MODEL_PROFILES = cast(ModelProfileRegistry, _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    """Get the default profile for a model.

    Args:
        model_name: The model identifier.

    Returns:
        The model profile dictionary, or an empty dict if not found.
    """
    default = _MODEL_PROFILES.get(model_name)
    if default:
        return default.copy()
    return {}


_FALLBACK_MAX_OUTPUT_TOKENS: Final[int] = 4096


class AnthropicTool(TypedDict):
    """Anthropic tool definition for custom (user-defined) tools.

    Custom tools use `name` and `input_schema` fields to define the tool's
    interface. These are converted from LangChain tool formats (functions, Pydantic
    models, `BaseTool` objects) via `convert_to_anthropic_tool`.
    """

    name: str

    input_schema: dict[str, Any]

    description: NotRequired[str]

    strict: NotRequired[bool]

    cache_control: NotRequired[dict[str, str]]

    defer_loading: NotRequired[bool]

    input_examples: NotRequired[list[dict[str, Any]]]

    allowed_callers: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Built-in Tool Support
# ---------------------------------------------------------------------------
# When Anthropic releases new built-in tools, two places may need updating:
#
# 1. _TOOL_TYPE_TO_BETA (below) - Add mapping if the tool requires a beta header.
#     Not all tools need this; only add if the API requires a beta header.
#
# 2. _is_builtin_tool() - Add the tool type prefix to _BUILTIN_TOOL_PREFIXES.
#     This ensures the tool dict is passed through to the API unchanged (instead
#     of being converted via convert_to_anthropic_tool, which may fail).
# ---------------------------------------------------------------------------

_TOOL_TYPE_TO_BETA: dict[str, str] = {
    "web_fetch_20250910": "web-fetch-2025-09-10",
    "code_execution_20250522": "code-execution-2025-05-22",
    "code_execution_20250825": "code-execution-2025-08-25",
    "mcp_toolset": "mcp-client-2025-11-20",
    "memory_20250818": "context-management-2025-06-27",
    "computer_20250124": "computer-use-2025-01-24",
    "computer_20251124": "computer-use-2025-11-24",
    "tool_search_tool_regex_20251119": "advanced-tool-use-2025-11-20",
    "tool_search_tool_bm25_20251119": "advanced-tool-use-2025-11-20",
}
"""Mapping of tool type to required beta header.

Some tool types require specific beta headers to be enabled.
"""

_BUILTIN_TOOL_PREFIXES = [
    "text_editor_",
    "computer_",
    "bash_",
    "web_search_",
    "web_fetch_",
    "code_execution_",
    "mcp_toolset",
    "memory_",
    "tool_search_",
]

_ANTHROPIC_EXTRA_FIELDS: set[str] = {
    "allowed_callers",
    "cache_control",
    "defer_loading",
    "input_examples",
}
"""Valid Anthropic-specific extra fields"""


def _is_builtin_tool(tool: Any) -> bool:
    """Check if a tool is a built-in (server-side) Anthropic tool.

    `tool` must be a `dict` and have a `type` key starting with one of the known
    built-in tool prefixes.

    [Claude docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
    """
    if not isinstance(tool, dict):
        return False

    tool_type = tool.get("type")
    if not tool_type or not isinstance(tool_type, str):
        return False

    return any(tool_type.startswith(prefix) for prefix in _BUILTIN_TOOL_PREFIXES)


def _format_image(url: str) -> dict:
    """Convert part["image_url"]["url"] strings (OpenAI format) to Anthropic format.

    {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "/9j/4AAQSkZJRg...",
    }

    Or

    {
        "type": "url",
        "url": "https://example.com/image.jpg",
    }
    """
    # Base64 encoded image
    base64_regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    base64_match = re.match(base64_regex, url)

    if base64_match:
        return {
            "type": "base64",
            "media_type": base64_match.group("media_type"),
            "data": base64_match.group("data"),
        }

    # Url
    url_regex = r"^https?://.*$"
    url_match = re.match(url_regex, url)

    if url_match:
        return {
            "type": "url",
            "url": url,
        }

    msg = (
        "Malformed url parameter."
        " Must be either an image URL (https://example.com/image.jpg)"
        " or base64 encoded string (data:image/png;base64,'/9j/4AAQSk'...)"
    )
    raise ValueError(
        msg,
    )


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> list[SystemMessage | AIMessage | HumanMessage]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        if isinstance(curr, ToolMessage):
            if (
                isinstance(curr.content, list)
                and curr.content
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in curr.content
                )
            ):
                curr = HumanMessage(curr.content)  # type: ignore[misc]
            else:
                curr = HumanMessage(  # type: ignore[misc]
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                            "is_error": curr.status == "error",
                        },
                    ],
                )
        last = merged[-1] if merged else None
        if any(
            all(isinstance(m, c) for m in (curr, last))
            for c in (SystemMessage, HumanMessage)
        ):
            if isinstance(cast("BaseMessage", last).content, str):
                new_content: list = [
                    {"type": "text", "text": cast("BaseMessage", last).content},
                ]
            else:
                new_content = copy.copy(cast("list", cast("BaseMessage", last).content))
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            merged[-1] = curr.model_copy(update={"content": new_content})
        else:
            merged.append(curr)
    return merged


def _format_data_content_block(block: dict) -> dict:
    """Format standard data content block to format expected by Anthropic."""
    if block["type"] == "image":
        if "url" in block:
            if block["url"].startswith("data:"):
                # Data URI
                formatted_block = {
                    "type": "image",
                    "source": _format_image(block["url"]),
                }
            else:
                formatted_block = {
                    "type": "image",
                    "source": {"type": "url", "url": block["url"]},
                }
        elif "base64" in block or block.get("source_type") == "base64":
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block["mime_type"],
                    "data": block.get("base64") or block.get("data", ""),
                },
            }
        elif "file_id" in block:
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": block["file_id"],
                },
            }
        elif block.get("source_type") == "id":
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": block["id"],
                },
            }
        else:
            msg = (
                "Anthropic only supports 'url', 'base64', or 'id' keys for image "
                "content blocks."
            )
            raise ValueError(
                msg,
            )

    elif block["type"] == "file":
        if "url" in block:
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": block["url"],
                },
            }
        elif "base64" in block or block.get("source_type") == "base64":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": block.get("mime_type") or "application/pdf",
                    "data": block.get("base64") or block.get("data", ""),
                },
            }
        elif block.get("source_type") == "text":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": block.get("mime_type") or "text/plain",
                    "data": block["text"],
                },
            }
        elif "file_id" in block:
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": block["file_id"],
                },
            }
        elif block.get("source_type") == "id":
            formatted_block = {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": block["id"],
                },
            }
        else:
            msg = (
                "Anthropic only supports 'url', 'base64', or 'id' keys for file "
                "content blocks."
            )
            raise ValueError(msg)

    elif block["type"] == "text-plain":
        formatted_block = {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": block.get("mime_type") or "text/plain",
                "data": block["text"],
            },
        }

    else:
        msg = f"Block of type {block['type']} is not supported."
        raise ValueError(msg)

    if formatted_block:
        for key in ["cache_control", "citations", "title", "context"]:
            if key in block:
                formatted_block[key] = block[key]
            elif (metadata := block.get("extras")) and key in metadata:
                formatted_block[key] = metadata[key]
            elif (metadata := block.get("metadata")) and key in metadata:
                # Backward compat
                formatted_block[key] = metadata[key]

    return formatted_block


def _format_messages(
    messages: Sequence[BaseMessage],
) -> tuple[str | list[dict] | None, list[dict]]:
    """Format messages for Anthropic's API."""
    system: str | list[dict] | None = None
    formatted_messages: list[dict] = []
    merged_messages = _merge_messages(messages)
    for _i, message in enumerate(merged_messages):
        if message.type == "system":
            if system is not None:
                msg = "Received multiple non-consecutive system messages."
                raise ValueError(msg)
            if isinstance(message.content, list):
                system = [
                    (
                        block
                        if isinstance(block, dict)
                        else {"type": "text", "text": block}
                    )
                    for block in message.content
                ]
            else:
                system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: str | list

        if not isinstance(message.content, str):
            # parse as dict
            if not isinstance(message.content, list):
                msg = "Anthropic message content must be str or list of dicts"
                raise ValueError(
                    msg,
                )

            # populate content
            content = []
            for block in message.content:
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif isinstance(block, dict):
                    if "type" not in block:
                        msg = "Dict content block must have a type key"
                        raise ValueError(msg)
                    if block["type"] == "image_url":
                        # convert format
                        source = _format_image(block["image_url"]["url"])
                        content.append({"type": "image", "source": source})
                    elif is_data_content_block(block):
                        content.append(_format_data_content_block(block))
                    elif block["type"] == "tool_use":
                        # If a tool_call with the same id as a tool_use content block
                        # exists, the tool_call is preferred.
                        if (
                            isinstance(message, AIMessage)
                            and (block["id"] in [tc["id"] for tc in message.tool_calls])
                            and not block.get("caller")
                        ):
                            overlapping = [
                                tc
                                for tc in message.tool_calls
                                if tc["id"] == block["id"]
                            ]
                            content.extend(
                                _lc_tool_calls_to_anthropic_tool_use_blocks(
                                    overlapping,
                                ),
                            )
                        else:
                            if tool_input := block.get("input"):
                                args = tool_input
                            elif "partial_json" in block:
                                try:
                                    args = json.loads(block["partial_json"] or "{}")
                                except json.JSONDecodeError:
                                    args = {}
                            else:
                                args = {}
                            tool_use_block = _AnthropicToolUse(
                                type="tool_use",
                                name=block["name"],
                                input=args,
                                id=block["id"],
                            )
                            if caller := block.get("caller"):
                                tool_use_block["caller"] = caller
                            content.append(tool_use_block)
                    elif block["type"] in ("server_tool_use", "mcp_tool_use"):
                        formatted_block = {
                            k: v
                            for k, v in block.items()
                            if k
                            in (
                                "type",
                                "id",
                                "input",
                                "name",
                                "server_name",  # for mcp_tool_use
                                "cache_control",
                            )
                        }
                        # Attempt to parse streamed output
                        if block.get("input") == {} and "partial_json" in block:
                            try:
                                input_ = json.loads(block["partial_json"])
                                if input_:
                                    formatted_block["input"] = input_
                            except json.JSONDecodeError:
                                pass
                        content.append(formatted_block)
                    elif block["type"] == "text":
                        text = block.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
                        if text.strip():
                            formatted_block = {
                                k: v
                                for k, v in block.items()
                                if k in ("type", "text", "cache_control", "citations")
                            }
                            # Clean up citations to remove null file_id fields
                            if formatted_block.get("citations"):
                                cleaned_citations = []
                                for citation in formatted_block["citations"]:
                                    cleaned_citation = {
                                        k: v
                                        for k, v in citation.items()
                                        if not (k == "file_id" and v is None)
                                    }
                                    cleaned_citations.append(cleaned_citation)
                                formatted_block["citations"] = cleaned_citations
                            content.append(formatted_block)
                    elif block["type"] == "thinking":
                        content.append(
                            {
                                k: v
                                for k, v in block.items()
                                if k
                                in ("type", "thinking", "cache_control", "signature")
                            },
                        )
                    elif block["type"] == "redacted_thinking":
                        content.append(
                            {
                                k: v
                                for k, v in block.items()
                                if k in ("type", "cache_control", "data")
                            },
                        )
                    elif (
                        block["type"] == "tool_result"
                        and isinstance(block.get("content"), list)
                        and any(
                            isinstance(item, dict)
                            and item.get("type") == "tool_reference"
                            for item in block["content"]
                        )
                    ):
                        # Tool search results with tool_reference blocks
                        content.append(
                            {
                                k: v
                                for k, v in block.items()
                                if k
                                in (
                                    "type",
                                    "content",
                                    "tool_use_id",
                                    "cache_control",
                                )
                            },
                        )
                    elif block["type"] == "tool_result":
                        # Regular tool results that need content formatting
                        tool_content = _format_messages(
                            [HumanMessage(block["content"])],
                        )[1][0]["content"]
                        content.append({**block, "content": tool_content})
                    elif block["type"] in (
                        "code_execution_tool_result",
                        "bash_code_execution_tool_result",
                        "text_editor_code_execution_tool_result",
                        "mcp_tool_result",
                        "web_search_tool_result",
                        "web_fetch_tool_result",
                    ):
                        content.append(
                            {
                                k: v
                                for k, v in block.items()
                                if k
                                in (
                                    "type",
                                    "content",
                                    "tool_use_id",
                                    "is_error",  # for mcp_tool_result
                                    "cache_control",
                                    "retrieved_at",  # for web_fetch_tool_result
                                )
                            },
                        )
                    else:
                        content.append(block)
                else:
                    msg = (
                        f"Content blocks must be str or dict, instead was: "
                        f"{type(block)}"
                    )
                    raise ValueError(
                        msg,
                    )
        else:
            content = message.content

        # Ensure all tool_calls have a tool_use content block
        if isinstance(message, AIMessage) and message.tool_calls:
            content = content or []
            content = (
                [{"type": "text", "text": message.content}]
                if isinstance(content, str) and content
                else content
            )
            tool_use_ids = [
                cast("dict", block)["id"]
                for block in content
                if cast("dict", block)["type"] == "tool_use"
            ]
            missing_tool_calls = [
                tc for tc in message.tool_calls if tc["id"] not in tool_use_ids
            ]
            cast("list", content).extend(
                _lc_tool_calls_to_anthropic_tool_use_blocks(missing_tool_calls),
            )

        if role == "assistant" and _i == len(merged_messages) - 1:
            if isinstance(content, str):
                content = content.rstrip()
            elif (
                isinstance(content, list)
                and content
                and isinstance(content[-1], dict)
                and content[-1].get("type") == "text"
            ):
                content[-1]["text"] = content[-1]["text"].rstrip()

        if not content and role == "assistant" and _i < len(merged_messages) - 1:
            # anthropic.BadRequestError: Error code: 400: all messages must have
            # non-empty content except for the optional final assistant message
            continue
        formatted_messages.append({"role": role, "content": content})
    return system, formatted_messages


def _collect_code_execution_tool_ids(formatted_messages: list[dict]) -> set[str]:
    """Collect tool_use IDs that were called by code_execution.

    These blocks cannot have cache_control applied per Anthropic API requirements.
    """
    code_execution_tool_ids: set[str] = set()

    for message in formatted_messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            caller = block.get("caller")
            if isinstance(caller, dict):
                caller_type = caller.get("type", "")
                if caller_type.startswith("code_execution"):
                    tool_id = block.get("id")
                    if tool_id:
                        code_execution_tool_ids.add(tool_id)

    return code_execution_tool_ids


def _is_code_execution_related_block(
    block: dict,
    code_execution_tool_ids: set[str],
) -> bool:
    """Check if a content block is related to code_execution.

    Returns True for blocks that should NOT have cache_control applied.
    """
    if not isinstance(block, dict):
        return False

    block_type = block.get("type")

    # tool_use blocks called by code_execution
    if block_type == "tool_use":
        caller = block.get("caller")
        if isinstance(caller, dict):
            caller_type = caller.get("type", "")
            if caller_type.startswith("code_execution"):
                return True

    # tool_result blocks for code_execution called tools
    if block_type == "tool_result":
        tool_use_id = block.get("tool_use_id")
        if tool_use_id and tool_use_id in code_execution_tool_ids:
            return True

    return False


class AnthropicContextOverflowError(anthropic.BadRequestError, ContextOverflowError):
    """BadRequestError raised when input exceeds Anthropic's context limit."""


def _handle_anthropic_bad_request(e: anthropic.BadRequestError) -> None:
    """Handle Anthropic BadRequestError."""
    if "prompt is too long" in e.message:
        raise AnthropicContextOverflowError(
            message=e.message, response=e.response, body=e.body
        ) from e
    if ("messages: at least one message is required") in e.message:
        message = "Received only system message(s). "
        warnings.warn(message, stacklevel=2)
        raise e
    raise


class ChatAnthropic(BaseChatModel):
    """Anthropic (Claude) chat models.

    See the [LangChain docs for `ChatAnthropic`](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
    for tutorials, feature walkthroughs, and examples.

    See the [Claude Platform docs](https://platform.claude.com/docs/en/about-claude/models/overview)
    for a list of the latest models, their capabilities, and pricing.

    Example:
        ```python
        # pip install -U langchain-anthropic
        # export ANTHROPIC_API_KEY="your-api-key"

        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            # temperature=,
            # max_tokens=,
            # timeout=,
            # max_retries=,
            # base_url="...",
            # Refer to API reference for full list of parameters
        )
        ```

    Note:
        Any param which is not explicitly supported will be passed directly to
        [`Anthropic.messages.create(...)`](https://platform.claude.com/docs/en/api/python/messages/create)
        each time to the model is invoked.
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int | None = Field(default=None, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation.

    If not specified, this is set dynamically using the model's `max_output_tokens`
    from its model profile.

    See docs on [model profiles](https://docs.langchain.com/oss/python/langchain/models#model-profiles)
    for more information.
    """

    temperature: float | None = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int | None = None
    """Number of most likely tokens to consider at each step."""

    top_p: float | None = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: float | None = Field(None, alias="timeout")
    """Timeout for requests to Claude API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 2
    """Number of retries allowed for requests sent to the Claude API."""

    stop_sequences: list[str] | None = Field(None, alias="stop")
    """Default stop sequences."""

    anthropic_api_url: str | None = Field(
        alias="base_url",
        default_factory=from_env(
            ["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"],
            default="https://api.anthropic.com",
        ),
    )
    """Base URL for API requests. Only specify if using a proxy or service emulator.

    If a value isn't passed in, will attempt to read the value first from
    `ANTHROPIC_API_URL` and if that is not set, `ANTHROPIC_BASE_URL`.
    """

    anthropic_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("ANTHROPIC_API_KEY", default=""),
    )
    """Automatically read from env var `ANTHROPIC_API_KEY` if not provided."""

    anthropic_proxy: str | None = Field(
        default_factory=from_env("ANTHROPIC_PROXY", default=None)
    )
    """Proxy to use for the Anthropic clients, will be used for every API call.

    If not provided, will attempt to read from the `ANTHROPIC_PROXY` environment
    variable.
    """

    default_headers: Mapping[str, str] | None = None
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    betas: list[str] | None = None
    """List of beta features to enable. If specified, invocations will be routed
    through `client.beta.messages.create`.

    Example: `#!python betas=["token-efficient-tools-2025-02-19"]`
    """
    # Can also be passed in w/ model_kwargs, but having it as a param makes better devx
    #
    # Precedence order:
    # 1. Call-time kwargs (e.g., llm.invoke(..., betas=[...]))
    # 2. model_kwargs (e.g., ChatAnthropic(model_kwargs={"betas": [...]}))
    # 3. Direct parameter (e.g., ChatAnthropic(betas=[...]))

    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output.

    If `True`, additional message chunks will be generated during the stream including
    usage metadata.
    """

    thinking: dict[str, Any] | None = Field(default=None)
    """Parameters for Claude reasoning,

    e.g., `#!python {"type": "enabled", "budget_tokens": 10_000}`

    For Claude Opus 4.6, `budget_tokens` is deprecated in favor of
    `#!python {"type": "adaptive"}`
    """

    effort: Literal["max", "high", "medium", "low"] | None = None
    """Control how many tokens Claude uses when responding.

    This parameter will be merged into the `output_config` parameter when making
    API calls.

    Example: `effort="medium"`

    !!! note

        Setting `effort` to `'high'` produces exactly the same behavior as omitting the
        parameter altogether.

    !!! note "Model Support"

        This feature is generally available on Claude Opus 4.6 and Claude Opus 4.5.
        The `max` effort level is only supported by Claude Opus 4.6.
    """

    mcp_servers: list[dict[str, Any]] | None = None
    """List of MCP servers to use for the request.

    Example: `#!python mcp_servers=[{"type": "url", "url": "https://mcp.example.com/mcp",
    "name": "example-mcp"}]`
    """

    context_management: dict[str, Any] | None = None
    """Configuration for
    [context management](https://platform.claude.com/docs/en/build-with-claude/context-editing).
    """

    reuse_last_container: bool | None = None
    """Automatically reuse container from most recent response (code execution).

    When using the built-in
    [code execution tool](https://docs.langchain.com/oss/python/integrations/chat/anthropic#code-execution),
    model responses will include container metadata. Set `reuse_last_container=True`
    to automatically reuse the container from the most recent response for subsequent
    invocations.
    """

    inference_geo: str | None = None
    """Controls where model inference runs. See Anthropic's
    [data residency](https://platform.claude.com/docs/en/build-with-claude/data-residency)
    docs for more information.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return a mapping of secret keys to environment variables."""
        return {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "mcp_servers": "ANTHROPIC_MCP_SERVERS",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Whether the class is serializable in langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "chat_models", "anthropic"]`
        """
        return ["langchain", "chat_models", "anthropic"]

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_kwargs": self.model_kwargs,
            "streaming": self.streaming,
            "max_retries": self.max_retries,
            "default_request_timeout": self.default_request_timeout,
            "thinking": self.thinking,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="anthropic",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    @model_validator(mode="before")
    @classmethod
    def set_default_max_tokens(cls, values: dict[str, Any]) -> Any:
        """Set default `max_tokens` from model profile with fallback."""
        if values.get("max_tokens") is None:
            model = values.get("model") or values.get("model_name")
            profile = _get_default_model_profile(model) if model else {}
            values["max_tokens"] = profile.get(
                "max_output_tokens", _FALLBACK_MAX_OUTPUT_TOKENS
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict) -> Any:
        """Build model kwargs."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model)
        return self

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        client_params: dict[str, Any] = {
            "api_key": self.anthropic_api_key.get_secret_value(),
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": (self.default_headers or None),
        }
        # value <= 0 indicates the param should be ignored. None is a meaningful value
        # for Anthropic client and treated differently than not specifying the param at
        # all.
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout

        return client_params

    @cached_property
    def _client(self) -> anthropic.Client:
        client_params = self._client_params
        http_client_params = {"base_url": client_params["base_url"]}
        if "timeout" in client_params:
            http_client_params["timeout"] = client_params["timeout"]
        if self.anthropic_proxy:
            http_client_params["anthropic_proxy"] = self.anthropic_proxy
        http_client = _get_default_httpx_client(**http_client_params)
        params = {
            **client_params,
            "http_client": http_client,
        }
        return anthropic.Client(**params)

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        client_params = self._client_params
        http_client_params = {"base_url": client_params["base_url"]}
        if "timeout" in client_params:
            http_client_params["timeout"] = client_params["timeout"]
        if self.anthropic_proxy:
            http_client_params["anthropic_proxy"] = self.anthropic_proxy
        http_client = _get_default_async_httpx_client(**http_client_params)
        params = {
            **client_params,
            "http_client": http_client,
        }
        return anthropic.AsyncClient(**params)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: dict,
    ) -> dict:
        """Get the request payload for the Anthropic API."""
        messages = self._convert_input(input_).to_messages()

        for idx, message in enumerate(messages):
            # Translate v1 content
            if (
                isinstance(message, AIMessage)
                and message.response_metadata.get("output_version") == "v1"
            ):
                tcs: list[types.ToolCall] = [
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                        "id": tool_call.get("id"),
                    }
                    for tool_call in message.tool_calls
                ]
                messages[idx] = message.model_copy(
                    update={
                        "content": _convert_from_v1_to_anthropic(
                            cast(list[types.ContentBlock], message.content),
                            tcs,
                            message.response_metadata.get("model_provider"),
                        )
                    }
                )

        system, formatted_messages = _format_messages(messages)

        # If cache_control is provided in kwargs, add it to the last eligible message
        # block (Anthropic requires cache_control to be nested within a message block).
        # Skip blocks related to code_execution as they cannot have cache_control.
        cache_control = kwargs.pop("cache_control", None)
        if cache_control and formatted_messages:
            # Collect tool IDs called by code_execution
            code_execution_tool_ids = _collect_code_execution_tool_ids(
                formatted_messages
            )

            cache_applied = False
            for formatted_message in reversed(formatted_messages):
                if cache_applied:
                    break
                content = formatted_message.get("content")
                if isinstance(content, list) and content:
                    # Find last eligible block (not code_execution related)
                    for block in reversed(content):
                        if isinstance(block, dict):
                            if _is_code_execution_related_block(
                                block, code_execution_tool_ids
                            ):
                                continue
                            block["cache_control"] = cache_control
                            cache_applied = True
                            break
                elif isinstance(content, str):
                    formatted_message["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]
                    cache_applied = True
            # If we didn't find an eligible block we silently drop the control.
            # Anthropic would reject a payload with cache_control on
            # code_execution blocks.
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop or self.stop_sequences,
            "betas": self.betas,
            "context_management": self.context_management,
            "mcp_servers": self.mcp_servers,
            "system": system,
            **self.model_kwargs,
            **kwargs,
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.inference_geo is not None:
            payload["inference_geo"] = self.inference_geo

        # Handle output_config and effort parameter
        # Priority: self.effort > payload output_config
        output_config = payload.get("output_config", {})
        output_config = output_config.copy() if isinstance(output_config, dict) else {}

        if self.effort:
            output_config["effort"] = self.effort

        if output_config:
            payload["output_config"] = output_config

        if "response_format" in payload:
            # response_format present when using agents.create_agent's ProviderStrategy
            # ---
            # ProviderStrategy converts to OpenAI-style format, which passes kwargs to
            # ChatAnthropic, ending up in our payload
            response_format = payload.pop("response_format")
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                response_format = cast(dict, response_format["json_schema"]["schema"])
            # Convert OpenAI-style response_format to Anthropic's output_config.format
            output_config = payload.setdefault("output_config", {})
            output_config["format"] = _convert_to_anthropic_output_config_format(
                response_format
            )

        # Handle deprecated output_format parameter for backward compatibility
        if "output_format" in payload:
            warnings.warn(
                "The 'output_format' parameter is deprecated and will be removed in a "
                "future version. Use 'output_config={\"format\": ...}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            output_config = payload.setdefault("output_config", {})
            output_config["format"] = payload.pop("output_format")

        if self.reuse_last_container:
            # Check for most recent AIMessage with container set in response_metadata
            # and set as a top-level param on the request
            for message in reversed(messages):
                if (
                    isinstance(message, AIMessage)
                    and (container := message.response_metadata.get("container"))
                    and isinstance(container, dict)
                    and (container_id := container.get("id"))
                ):
                    payload["container"] = container_id
                    break

        # Note: Beta headers are no longer required for structured outputs
        # (output_config.format or strict tool use) as they are now generally available
        if "tools" in payload and isinstance(payload["tools"], list):
            # Auto-append required betas for specific tool types and input_examples
            has_input_examples = False
            for tool in payload["tools"]:
                if isinstance(tool, dict):
                    tool_type = tool.get("type")
                    if tool_type and tool_type in _TOOL_TYPE_TO_BETA:
                        required_beta = _TOOL_TYPE_TO_BETA[tool_type]
                        if payload["betas"]:
                            if required_beta not in payload["betas"]:
                                payload["betas"] = [
                                    *payload["betas"],
                                    required_beta,
                                ]
                        else:
                            payload["betas"] = [required_beta]
                    # Check for input_examples
                    if tool.get("input_examples"):
                        has_input_examples = True

            # Auto-append header for input_examples
            if has_input_examples:
                required_beta = "advanced-tool-use-2025-11-20"
                if payload["betas"]:
                    if required_beta not in payload["betas"]:
                        payload["betas"] = [*payload["betas"], required_beta]
                else:
                    payload["betas"] = [required_beta]

        # Auto-append required beta for mcp_servers
        if payload.get("mcp_servers"):
            required_beta = "mcp-client-2025-11-20"
            if payload["betas"]:
                # Append to existing betas if not already present
                if required_beta not in payload["betas"]:
                    payload["betas"] = [*payload["betas"], required_beta]
            else:
                payload["betas"] = [required_beta]

        return {k: v for k, v in payload.items() if v is not None}

    def _create(self, payload: dict) -> Any:
        if "betas" in payload:
            return self._client.beta.messages.create(**payload)
        return self._client.messages.create(**payload)

    async def _acreate(self, payload: dict) -> Any:
        if "betas" in payload:
            return await self._async_client.beta.messages.create(**payload)
        return await self._async_client.messages.create(**payload)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            stream = self._create(payload)
            coerce_content_to_string = (
                not _tools_in_params(payload)
                and not _documents_in_params(payload)
                and not _thinking_in_params(payload)
                and not _compact_in_params(payload)
            )
            block_start_event = None
            for event in stream:
                msg, block_start_event = _make_message_chunk_from_anthropic_event(
                    event,
                    stream_usage=stream_usage,
                    coerce_content_to_string=coerce_content_to_string,
                    block_start_event=block_start_event,
                )
                if msg is not None:
                    chunk = ChatGenerationChunk(message=msg)
                    if run_manager and isinstance(msg.content, str):
                        run_manager.on_llm_new_token(msg.content, chunk=chunk)
                    yield chunk
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            stream = await self._acreate(payload)
            coerce_content_to_string = (
                not _tools_in_params(payload)
                and not _documents_in_params(payload)
                and not _thinking_in_params(payload)
                and not _compact_in_params(payload)
            )
            block_start_event = None
            async for event in stream:
                msg, block_start_event = _make_message_chunk_from_anthropic_event(
                    event,
                    stream_usage=stream_usage,
                    coerce_content_to_string=coerce_content_to_string,
                    block_start_event=block_start_event,
                )
                if msg is not None:
                    chunk = ChatGenerationChunk(message=msg)
                    if run_manager and isinstance(msg.content, str):
                        await run_manager.on_llm_new_token(msg.content, chunk=chunk)
                    yield chunk
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        """Format the output from the Anthropic API to LC."""
        data_dict = data.model_dump()
        content = data_dict["content"]

        # Remove citations if they are None - introduced in anthropic sdk 0.45
        for block in content:
            if isinstance(block, dict):
                if "citations" in block and block["citations"] is None:
                    block.pop("citations")
                if "caller" in block and block["caller"] is None:
                    block.pop("caller")
                if (
                    block.get("type") == "thinking"
                    and "text" in block
                    and block["text"] is None
                ):
                    block.pop("text")

        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if (
            (container := llm_output.get("container"))
            and isinstance(container, dict)
            and (expires_at := container.get("expires_at"))
            and isinstance(expires_at, datetime.datetime)
        ):
            # TODO: dump all `data` with `mode="json"`
            llm_output["container"]["expires_at"] = expires_at.isoformat()
        response_metadata = {"model_provider": "anthropic"}
        if "model" in llm_output and "model_name" not in llm_output:
            llm_output["model_name"] = llm_output["model"]
        if (
            len(content) == 1
            and content[0]["type"] == "text"
            and not content[0].get("citations")
        ):
            msg = AIMessage(
                content=content[0]["text"], response_metadata=response_metadata
            )
        elif any(block["type"] == "tool_use" for block in content):
            tool_calls = extract_tool_calls(content)
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata=response_metadata,
            )
        else:
            msg = AIMessage(content=content, response_metadata=response_metadata)
        msg.usage_metadata = _create_usage_metadata(data.usage)
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            data = self._create(payload)
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        return self._format_output(data, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            data = await self._acreate(payload)
        except anthropic.BadRequestError as e:
            _handle_anthropic_bad_request(e)
        return self._format_output(data, **kwargs)

    def _get_llm_for_structured_output_when_thinking_is_enabled(
        self,
        schema: dict | type,
        formatted_tool: AnthropicTool,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        thinking_admonition = (
            "You are attempting to use structured output via forced tool calling, "
            "which is not guaranteed when `thinking` is enabled. This method will "
            "raise an OutputParserException if tool calls are not generated. Consider "
            "disabling `thinking` or adjust your prompt to ensure the tool is called."
        )
        warnings.warn(thinking_admonition, stacklevel=2)
        llm = self.bind_tools(
            [schema],
            # We don't specify tool_choice here since the API will reject attempts to
            # force tool calls when thinking=true
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": formatted_tool,
            },
        )

        def _raise_if_no_tool_calls(message: AIMessage) -> AIMessage:
            if not message.tool_calls:
                raise OutputParserException(thinking_admonition)
            return message

        return llm | _raise_if_no_tool_calls

    def bind_tools(
        self,
        tools: Sequence[Mapping[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict[str, str] | str | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        r"""Bind tool-like objects to `ChatAnthropic`.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Supports Anthropic format tool schemas and any tool definition handled
                by [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: Which tool to require the model to call. Options are:

                - Name of the tool as a string or as dict `{"type": "tool", "name": "<<tool_name>>"}`: calls corresponding tool
                - `'auto'`, `{"type: "auto"}`, or `None`: automatically selects a tool (including no tool)
                - `'any'` or `{"type: "any"}`: force at least one tool to be called
            parallel_tool_calls: Set to `False` to disable parallel tool use.

                Defaults to `None` (no specification, which allows parallel tool use).

                !!! version-added "Added in `langchain-anthropic` 0.3.2"
            strict: If `True`, Claude's schema adherence is applied to tool calls.

                See the [docs](https://docs.langchain.com/oss/python/integrations/chat/anthropic#strict-tool-use) for more info.
            kwargs: Any additional parameters are passed directly to `bind`.

        Example:
            ```python
            from langchain_anthropic import ChatAnthropic
            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            class GetPrice(BaseModel):
                '''Get the price of a specific product.'''

                product: str = Field(..., description="The product to look up.")


            model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
            model_with_tools = model.bind_tools([GetWeather, GetPrice])
            model_with_tools.invoke(
                "What is the weather like in San Francisco",
            )
            # -> AIMessage(
            #     content=[
            #         {'text': '<thinking>\nBased on the user\'s question, the relevant function to call is GetWeather, which requires the "location" parameter.\n\nThe user has directly specified the location as "San Francisco". Since San Francisco is a well known city, I can reasonably infer they mean San Francisco, CA without needing the state specified.\n\nAll the required parameters are provided, so I can proceed with the API call.\n</thinking>', 'type': 'text'},
            #         {'text': None, 'type': 'tool_use', 'id': 'toolu_01SCgExKzQ7eqSkMHfygvYuu', 'name': 'GetWeather', 'input': {'location': 'San Francisco, CA'}}
            #     ],
            #     response_metadata={'id': 'msg_01GM3zQtoFv8jGQMW7abLnhi', 'model': 'claude-sonnet-4-5-20250929', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 487, 'output_tokens': 145}},
            #     id='run-87b1331e-9251-4a68-acef-f0a018b639cc-0'
            # )
            ```
        """  # noqa: E501
        # Allows built-in tools either by their:
        # - Raw `dict` format
        # - Extracting extras["provider_tool_definition"] if provided on a BaseTool
        formatted_tools = [
            tool
            if _is_builtin_tool(tool)
            else convert_to_anthropic_tool(tool, strict=strict)
            for tool in tools
        ]
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            msg = (
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, "
                f"str, or None."
            )
            raise ValueError(
                msg,
            )

        if parallel_tool_calls is not None:
            disable_parallel_tool_use = not parallel_tool_calls
            if "tool_choice" in kwargs:
                kwargs["tool_choice"]["disable_parallel_tool_use"] = (
                    disable_parallel_tool_use
                )
            else:
                kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": disable_parallel_tool_use,
                }

        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        method: Literal["function_calling", "json_schema"] = "function_calling",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        See the [LangChain docs](https://docs.langchain.com/oss/python/integrations/chat/anthropic#structured-output)
        for more details and examples.

        Args:
            schema: The output schema. Can be passed in as:

                - An Anthropic tool schema,
                - An OpenAI function/tool schema,
                - A JSON Schema,
                - A `TypedDict` class,
                - Or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.
            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            method: The structured output method to use. Options are:

                - `'function_calling'` (default): Use forced tool calling to get
                    structured output.
                - `'json_schema'`: Use Claude's dedicated
                    [structured output](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
                    feature.

            kwargs: Additional keyword arguments are ignored.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`.

                If `include_raw` is `False` and `schema` is a Pydantic class, `Runnable`
                outputs an instance of `schema` (i.e., a Pydantic object). Otherwise, if
                `include_raw` is `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        Example:
            ```python hl_lines="13"
            from langchain_anthropic import ChatAnthropic
            from pydantic import BaseModel, Field

            model = ChatAnthropic(model="claude-sonnet-4-5")

            class Movie(BaseModel):
                \"\"\"A movie with details.\"\"\"
                title: str = Field(..., description="The title of the movie")
                year: int = Field(..., description="The year the movie was released")
                director: str = Field(..., description="The director of the movie")
                rating: float = Field(..., description="The movie's rating out of 10")

            model_with_structure = model.with_structured_output(Movie, method="json_schema")
            response = model_with_structure.invoke("Provide details about the movie Inception")
            print(response)
            # -> Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
            ```
        """  # noqa: E501
        if method == "json_mode":
            warning_message = (
                "Unrecognized structured output method 'json_mode'. Defaulting to "
                "'json_schema' method."
            )
            warnings.warn(warning_message, stacklevel=2)
            method = "json_schema"

        if method == "function_calling":
            formatted_tool = cast(AnthropicTool, convert_to_anthropic_tool(schema))
            # The result of convert_to_anthropic_tool for 'method=function_calling' will
            # always be an AnthropicTool
            tool_name = formatted_tool["name"]
            if self.thinking is not None and self.thinking.get("type") == "enabled":
                llm = self._get_llm_for_structured_output_when_thinking_is_enabled(
                    schema,
                    formatted_tool,
                )
            else:
                llm = self.bind_tools(
                    [schema],
                    tool_choice=tool_name,  # Force tool call
                    ls_structured_output_format={
                        "kwargs": {"method": "function_calling"},
                        "schema": formatted_tool,
                    },
                )

            if isinstance(schema, type) and is_basemodel_subclass(schema):
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name,
                    first_tool_only=True,
                )
        elif method == "json_schema":
            llm = self.bind(
                output_config={
                    "format": _convert_to_anthropic_output_config_format(schema)
                },
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            error_message = (
                f"Unrecognized structured output method '{method}'. "
                f"Expected 'function_calling' or 'json_schema'."
            )
            raise ValueError(error_message)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none],
                exception_key="parsing_error",
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
        **kwargs: Any,
    ) -> int:
        """Count tokens in a sequence of input messages.

        This uses Anthropic's official [token counting API](https://platform.claude.com/docs/en/build-with-claude/token-counting).

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of `dict`, `BaseModel`, function, or `BaseTool`
                objects to be converted to tool schemas.
            kwargs: Additional keyword arguments are passed to the Anthropic
                `messages.count_tokens` method.

        ???+ example "Basic usage"

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage

            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

            messages = [
                SystemMessage(content="You are a scientist"),
                HumanMessage(content="Hello, Claude"),
            ]
            model.get_num_tokens_from_messages(messages)
            ```

            ```txt
            14
            ```

        ??? example "Pass tool schemas"

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            from langchain_core.tools import tool

            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

            @tool(parse_docstring=True)
            def get_weather(location: str) -> str:
                \"\"\"Get the current weather in a given location

                Args:
                    location: The city and state, e.g. San Francisco, CA
                \"\"\"
                return "Sunny"

            messages = [
                HumanMessage(content="What's the weather like in San Francisco?"),
            ]
            model.get_num_tokens_from_messages(messages, tools=[get_weather])
            ```

            ```txt
            403
            ```
        """  # noqa: D214
        formatted_system, formatted_messages = _format_messages(messages)
        if isinstance(formatted_system, str):
            kwargs["system"] = formatted_system
        if tools:
            kwargs["tools"] = [convert_to_anthropic_tool(tool) for tool in tools]
        if self.context_management is not None:
            kwargs["context_management"] = self.context_management

        if self.betas is not None:
            beta_response = self._client.beta.messages.count_tokens(
                betas=self.betas,
                model=self.model,
                messages=formatted_messages,  # type: ignore[arg-type]
                **kwargs,
            )
            return beta_response.input_tokens
        response = self._client.messages.count_tokens(
            model=self.model,
            messages=formatted_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.input_tokens


def convert_to_anthropic_tool(
    tool: Mapping[str, Any] | type | Callable | BaseTool,
    *,
    strict: bool | None = None,
) -> AnthropicTool:
    """Convert a tool-like object to an Anthropic tool definition.

    Args:
        tool: A tool-like object to convert. Can be an Anthropic tool dict,
            a Pydantic model, a function, or a `BaseTool`.
        strict: If `True`, enables strict schema adherence for the tool.

            !!! note

                Requires Claude Sonnet 4.5 or Opus 4.1.

    Returns:
        `AnthropicTool` for custom/user-defined tools
    """
    if (
        isinstance(tool, BaseTool)
        and hasattr(tool, "extras")
        and isinstance(tool.extras, dict)
        and "provider_tool_definition" in tool.extras
    ):
        # Pass through built-in tool definitions
        return tool.extras["provider_tool_definition"]  # type: ignore[return-value]

    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        # Anthropic tool format
        anthropic_formatted = AnthropicTool(tool)  # type: ignore[misc]
    else:
        oai_formatted = convert_to_openai_tool(tool, strict=strict)["function"]
        anthropic_formatted = AnthropicTool(
            name=oai_formatted["name"],
            input_schema=oai_formatted["parameters"],
        )
        if "description" in oai_formatted:
            anthropic_formatted["description"] = oai_formatted["description"]
        if "strict" in oai_formatted and isinstance(strict, bool):
            anthropic_formatted["strict"] = oai_formatted["strict"]
        # Select params from tool.extras
        if (
            isinstance(tool, BaseTool)
            and hasattr(tool, "extras")
            and isinstance(tool.extras, dict)
        ):
            for key, value in tool.extras.items():
                if key in _ANTHROPIC_EXTRA_FIELDS:
                    # all are populated top-level
                    anthropic_formatted[key] = value  # type: ignore[literal-required]
    return anthropic_formatted


def _tools_in_params(params: dict) -> bool:
    return (
        "tools" in params
        or ("extra_body" in params and params["extra_body"].get("tools"))
        or "mcp_servers" in params
    )


def _thinking_in_params(params: dict) -> bool:
    return params.get("thinking", {}).get("type") == "enabled"


def _documents_in_params(params: dict) -> bool:
    for message in params.get("messages", []):
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "document"
                    and block.get("citations", {}).get("enabled")
                ):
                    return True
    return False


def _compact_in_params(params: dict) -> bool:
    edits = params.get("context_management", {}).get("edits") or []

    return any("compact" in (edit.get("type") or "") for edit in edits)


class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str
    caller: NotRequired[dict[str, Any]]


def _lc_tool_calls_to_anthropic_tool_use_blocks(
    tool_calls: list[ToolCall],
) -> list[_AnthropicToolUse]:
    return [
        _AnthropicToolUse(
            type="tool_use",
            name=tool_call["name"],
            input=tool_call["args"],
            id=cast("str", tool_call["id"]),
        )
        for tool_call in tool_calls
    ]


def _convert_to_anthropic_output_config_format(schema: dict | type) -> dict[str, Any]:
    """Convert JSON schema, Pydantic model, or `TypedDict` into `output_config.format`.

    See Claude docs on [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

    Args:
        schema: A JSON schema dict, Pydantic model class, or TypedDict.

    Returns:
        A dict with `type` and `schema` keys suitable for `output_config.format`.
    """
    from anthropic import transform_schema

    is_pydantic_class = isinstance(schema, type) and is_basemodel_subclass(schema)
    if is_pydantic_class or isinstance(schema, dict):
        json_schema = transform_schema(schema)
    else:
        # TypedDict
        json_schema = transform_schema(convert_to_json_schema(schema))
    return {"type": "json_schema", "schema": json_schema}


def _make_message_chunk_from_anthropic_event(
    event: anthropic.types.RawMessageStreamEvent,
    *,
    stream_usage: bool = True,
    coerce_content_to_string: bool,
    block_start_event: anthropic.types.RawMessageStreamEvent | None = None,
) -> tuple[AIMessageChunk | None, anthropic.types.RawMessageStreamEvent | None]:
    """Convert Anthropic streaming event to `AIMessageChunk`.

    Args:
        event: Raw streaming event from Anthropic SDK
        stream_usage: Whether to include usage metadata in the output chunks.
        coerce_content_to_string: Whether to convert structured content to plain
            text strings.

            When `True`, only text content is preserved; when `False`, structured
            content like tool calls and citations are maintained.
        block_start_event: Previous content block start event, used for tracking
            tool use blocks and maintaining context across related events.

    Returns:
        Tuple with
            - `AIMessageChunk`: Converted message chunk with appropriate content and
                metadata, or `None` if the event doesn't produce a chunk
            - `RawMessageStreamEvent`: Updated `block_start_event` for tracking content
                blocks across sequential events, or `None` if not applicable

    Note:
        Not all Anthropic events result in message chunks. Events like internal
        state changes return `None` for the message chunk while potentially
        updating the `block_start_event` for context tracking.
    """
    message_chunk: AIMessageChunk | None = None
    # Reference: Anthropic SDK streaming implementation
    # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/streaming/_messages.py  # noqa: E501
    if event.type == "message_start" and stream_usage:
        # Capture model name, but don't include usage_metadata yet
        # as it will be properly reported in message_delta with complete info
        if hasattr(event.message, "model"):
            response_metadata: dict[str, Any] = {"model_name": event.message.model}
        else:
            response_metadata = {}

        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            response_metadata=response_metadata,
        )

    elif (
        event.type == "content_block_start"
        and event.content_block is not None
        and (
            "tool_result" in event.content_block.type
            or "tool_use" in event.content_block.type
            or "document" in event.content_block.type
            or "redacted_thinking" in event.content_block.type
        )
    ):
        if coerce_content_to_string:
            warnings.warn("Received unexpected tool content block.", stacklevel=2)

        content_block = event.content_block.model_dump()
        if "caller" in content_block and content_block["caller"] is None:
            content_block.pop("caller")
        content_block["index"] = event.index
        if event.content_block.type == "tool_use":
            if (
                parsed_args := getattr(event.content_block, "input", None)
            ) and isinstance(parsed_args, dict):
                # In some cases parsed args are represented in start event, with no
                # following input_json_delta events
                args = json.dumps(parsed_args)
            else:
                args = ""
            tool_call_chunk = create_tool_call_chunk(
                index=event.index,
                id=event.content_block.id,
                name=event.content_block.name,
                args=args,
            )
            tool_call_chunks = [tool_call_chunk]
        else:
            tool_call_chunks = []
        message_chunk = AIMessageChunk(
            content=[content_block],
            tool_call_chunks=tool_call_chunks,
        )
        block_start_event = event

    # Process incremental content updates
    elif event.type == "content_block_delta":
        # Text and citation deltas (incremental text content)
        if event.delta.type in ("text_delta", "citations_delta"):
            if coerce_content_to_string and hasattr(event.delta, "text"):
                text = getattr(event.delta, "text", "")
                message_chunk = AIMessageChunk(content=text)
            else:
                content_block = event.delta.model_dump()
                content_block["index"] = event.index

                # All citation deltas are part of a text block
                content_block["type"] = "text"
                if "citation" in content_block:
                    # Assign citations to a list if present
                    content_block["citations"] = [content_block.pop("citation")]
                message_chunk = AIMessageChunk(content=[content_block])

        # Reasoning
        elif event.delta.type in {"thinking_delta", "signature_delta"}:
            content_block = event.delta.model_dump()
            content_block["index"] = event.index
            content_block["type"] = "thinking"
            message_chunk = AIMessageChunk(content=[content_block])

        # Tool input JSON (streaming tool arguments)
        elif event.delta.type == "input_json_delta":
            content_block = event.delta.model_dump()
            content_block["index"] = event.index
            start_event_block = (
                getattr(block_start_event, "content_block", None)
                if block_start_event
                else None
            )
            if (
                start_event_block is not None
                and getattr(start_event_block, "type", None) == "tool_use"
            ):
                tool_call_chunk = create_tool_call_chunk(
                    index=event.index,
                    id=None,
                    name=None,
                    args=event.delta.partial_json,
                )
                tool_call_chunks = [tool_call_chunk]
            else:
                tool_call_chunks = []
            message_chunk = AIMessageChunk(
                content=[content_block],
                tool_call_chunks=tool_call_chunks,
            )

        # Compaction block
        elif event.delta.type == "compaction_delta":
            content_block = event.delta.model_dump()
            content_block["index"] = event.index
            content_block["type"] = "compaction"
            message_chunk = AIMessageChunk(content=[content_block])

    # Process final usage metadata and completion info
    elif event.type == "message_delta" and stream_usage:
        usage_metadata = _create_usage_metadata(event.usage)
        response_metadata = {
            "stop_reason": event.delta.stop_reason,
            "stop_sequence": event.delta.stop_sequence,
        }
        if context_management := getattr(event, "context_management", None):
            response_metadata["context_management"] = context_management.model_dump()
        message_delta = getattr(event, "delta", None)
        if message_delta and (container := getattr(message_delta, "container", None)):
            response_metadata["container"] = container.model_dump(mode="json")
        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        if message_chunk.response_metadata.get("stop_reason"):
            # Mark final Anthropic stream chunk
            message_chunk.chunk_position = "last"
    # Unhandled event types (e.g., `content_block_stop`, `ping` events)
    # https://platform.claude.com/docs/en/build-with-claude/streaming#other-events
    else:
        pass

    if message_chunk:
        message_chunk.response_metadata["model_provider"] = "anthropic"
    return message_chunk, block_start_event


def _create_usage_metadata(anthropic_usage: BaseModel) -> UsageMetadata:
    """Create LangChain `UsageMetadata` from Anthropic `Usage` data.

    Note:
        Anthropic's `input_tokens` excludes cached tokens, so we manually add
        `cache_read` and `cache_creation` tokens to get the true total.
    """
    input_token_details: dict = {
        "cache_read": getattr(anthropic_usage, "cache_read_input_tokens", None),
        "cache_creation": getattr(anthropic_usage, "cache_creation_input_tokens", None),
    }

    # Add cache TTL information if provided (5-minute and 1-hour ephemeral cache)
    cache_creation = getattr(anthropic_usage, "cache_creation", None)

    # Currently just copying over the 5m and 1h keys, but if more are added in the
    # future we'll need to expand this tuple
    cache_creation_keys = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    if cache_creation:
        if isinstance(cache_creation, BaseModel):
            cache_creation = cache_creation.model_dump()
        for k in cache_creation_keys:
            input_token_details[k] = cache_creation.get(k)

    # Calculate total input tokens: Anthropic's `input_tokens` excludes cached tokens,
    # so we need to add them back to get the true total input token count
    input_tokens = (
        (getattr(anthropic_usage, "input_tokens", 0) or 0)  # Base input tokens
        + (input_token_details["cache_read"] or 0)  # Tokens read from cache
        + (input_token_details["cache_creation"] or 0)  # Tokens used to create cache
    )
    output_tokens = getattr(anthropic_usage, "output_tokens", 0) or 0

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_token_details=InputTokenDetails(
            **{k: v for k, v in input_token_details.items() if v is not None},
        ),
    )
