"""Anthropic chat models."""

from __future__ import annotations

import copy
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
from langchain_core.exceptions import OutputParserException
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
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


_MODEL_DEFAULT_MAX_OUTPUT_TOKENS: Final[dict[str, int]] = {
    # Listed old to new
    "claude-3-haiku": 4096,  # Claude Haiku 3
    "claude-3-5-haiku": 8192,  # Claude Haiku 3.5
    "claude-3-7-sonnet": 64000,  # Claude Sonnet 3.7
    "claude-sonnet-4": 64000,  # Claude Sonnet 4
    "claude-opus-4": 32000,  # Claude Opus 4
    "claude-opus-4-1": 32000,  # Claude Opus 4.1
    "claude-sonnet-4-5": 64000,  # Claude Sonnet 4.5
    "claude-haiku-4-5": 64000,  # Claude Haiku 4.5
}
_FALLBACK_MAX_OUTPUT_TOKENS: Final[int] = 4096


def _default_max_tokens_for(model: str | None) -> int:
    """Return the default max output tokens for an Anthropic model (with fallback).

    See the Claude docs for [Max Tokens limits](https://docs.claude.com/en/docs/about-claude/models/overview#model-comparison-table).
    """
    if not model:
        return _FALLBACK_MAX_OUTPUT_TOKENS

    parts = model.split("-")
    family = "-".join(parts[:-1]) if len(parts) > 1 else model

    return _MODEL_DEFAULT_MAX_OUTPUT_TOKENS.get(family, _FALLBACK_MAX_OUTPUT_TOKENS)


class AnthropicTool(TypedDict):
    """Anthropic tool definition."""

    name: str

    input_schema: dict[str, Any]

    description: NotRequired[str]

    strict: NotRequired[bool]

    cache_control: NotRequired[dict[str, str]]


def _is_builtin_tool(tool: Any) -> bool:
    """Check if a tool is a built-in Anthropic tool.

    [Claude docs for built-in tools](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)
    """
    if not isinstance(tool, dict):
        return False

    tool_type = tool.get("type")
    if not tool_type or not isinstance(tool_type, str):
        return False

    _builtin_tool_prefixes = [
        "text_editor_",
        "computer_",
        "bash_",
        "web_search_",
        "web_fetch_",
        "code_execution_",
        "memory_",
    ]
    return any(tool_type.startswith(prefix) for prefix in _builtin_tool_prefixes)


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
                        if isinstance(message, AIMessage) and block["id"] in [
                            tc["id"] for tc in message.tool_calls
                        ]:
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
                            content.append(
                                _AnthropicToolUse(
                                    type="tool_use",
                                    name=block["name"],
                                    input=args,
                                    id=block["id"],
                                )
                            )
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
                    elif block["type"] == "tool_result":
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

        if not content and role == "assistant" and _i < len(merged_messages) - 1:
            # anthropic.BadRequestError: Error code: 400: all messages must have
            # non-empty content except for the optional final assistant message
            continue
        formatted_messages.append({"role": role, "content": content})
    return system, formatted_messages


def _handle_anthropic_bad_request(e: anthropic.BadRequestError) -> None:
    """Handle Anthropic BadRequestError."""
    if ("messages: at least one message is required") in e.message:
        message = "Received only system message(s). "
        warnings.warn(message, stacklevel=2)
        raise e
    raise


class ChatAnthropic(BaseChatModel):
    """Anthropic chat models.

    See [Anthropic's docs](https://docs.claude.com/en/docs/about-claude/models/overview)
    for a list of the latest models.

    Setup:
        Install `langchain-anthropic` and set environment variable `ANTHROPIC_API_KEY`.

        ```bash
        pip install -U langchain-anthropic
        export ANTHROPIC_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Anthropic model to use. e.g. `'claude-sonnet-4-5-20250929'`.
        temperature:
            Sampling temperature. Ranges from `0.0` to `1.0`.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        anthropic_proxy:
            Proxy to use for the Anthropic clients, will be used for every API call.
            If not passed in will be read from env var `ANTHROPIC_PROXY`.
        max_retries:
            Max number of retries if a request fails.
        api_key:
            Anthropic API key. If not passed in will be read from env var
            `ANTHROPIC_API_KEY`.
        base_url:
            Base URL for API requests. Only specify if using a proxy or service
            emulator.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # other params...
        )
        ```

    !!! note
        Any param which is not explicitly supported will be passed directly to the
        `anthropic.Anthropic.messages.create(...)` API every time to the model is
        invoked. For example:

        ```python
        from langchain_anthropic import ChatAnthropic
        import anthropic

        ChatAnthropic(..., extra_headers={}).invoke(...)

        # results in underlying API call of:

        anthropic.Anthropic(..).messages.create(..., extra_headers={})

        # which is also equivalent to:

        ChatAnthropic(...).invoke(..., extra_headers={})
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        ```python
        AIMessage(
            content="J'aime la programmation.",
            response_metadata={
                "id": "msg_01Trik66aiQ9Z1higrD5XFx3",
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 25, "output_tokens": 11},
            },
            id="run-5886ac5f-3c2e-49f5-8a44-b1e92808c929-0",
            usage_metadata={
                "input_tokens": 25,
                "output_tokens": 11,
                "total_tokens": 36,
            },
        )
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        ```python
        AIMessageChunk(content="J", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content="'", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content="a", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content="ime", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content=" la", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content=" programm", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content="ation", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        AIMessageChunk(content=".", id="run-272ff5f9-8485-402c-b90d-eac8babc5b25")
        ```

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(content="J'aime la programmation.", id="run-b34faef0-882f-4869-a19c-ed2b856e6361")
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

        ```python
        AIMessage(
            content="J'aime la programmation.",
            response_metadata={
                "id": "msg_01Trik66aiQ9Z1higrD5XFx3",
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 25, "output_tokens": 11},
            },
            id="run-5886ac5f-3c2e-49f5-8a44-b1e92808c929-0",
            usage_metadata={
                "input_tokens": 25,
                "output_tokens": 11,
                "total_tokens": 36,
            },
        )
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "toolu_01KzpPEAgzura7hpBqwHbWdo",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York, NY"},
                "id": "toolu_01JtgbVGVJbiSwtZk3Uycezx",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "toolu_01429aygngesudV9nTbCKGuw",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York, NY"},
                "id": "toolu_01JPktyd44tVMeBcPPnFSEJG",
            },
        ]
        ```

        See `ChatAnthropic.bind_tools()` method for more.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why was the cat sitting on the computer?",
            punchline="To keep an eye on the mouse!",
            rating=None,
        )
        ```

        See `ChatAnthropic.with_structured_output()` for more.

    Image input:
        See [multimodal guides](https://docs.langchain.com/oss/python/langchain/models#multimodal)
        for more detail.

        ```python
        import base64

        import httpx
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Can you highlight the differences between these two images?",
                },
                {
                    "type": "image",
                    "base64": image_data,
                    "mime_type": "image/jpeg",
                },
                {
                    "type": "image",
                    "url": image_url,
                },
            ],
        )
        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```python
        "After examining both images carefully, I can see that they are actually identical."
        ```

        ??? note "Files API"

            You can also pass in files that are managed through Anthropic's
            [Files API](https://docs.claude.com/en/docs/build-with-claude/files):

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                betas=["files-api-2025-04-14"],
            )
            input_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this document.",
                    },
                    {
                        "type": "image",
                        "id": "file_abc123...",
                    },
                ],
            }
            model.invoke([input_message])
            ```

    PDF input:
        See [multimodal guides](https://docs.langchain.com/oss/python/langchain/models#multimodal)
        for more detail.

        ```python
        from base64 import b64encode
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        import requests

        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        data = b64encode(requests.get(url).content).decode()

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
        ai_msg = model.invoke(
            [
                HumanMessage(
                    [
                        "Summarize this document.",
                        {
                            "type": "file",
                            "mime_type": "application/pdf",
                            "base64": data,
                        },
                    ]
                )
            ]
        )
        ai_msg.content
        ```

        ```python
        "This appears to be a simple document..."
        ```

        ??? note "Files API"

            You can also pass in files that are managed through Anthropic's
            [Files API](https://docs.claude.com/en/docs/build-with-claude/files):

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                betas=["files-api-2025-04-14"],
            )
            input_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this document.",
                    },
                    {
                        "type": "file",
                        "id": "file_abc123...",
                    },
                ],
            }
            model.invoke([input_message])
            ```

    Extended thinking:
        Certain [Claude models](https://docs.claude.com/en/docs/build-with-claude/extended-thinking#supported-models)
        support an [extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)
        feature, which will output the step-by-step reasoning process that led to its
        final answer.

        To use it, specify the `thinking` parameter when initializing `ChatAnthropic`.

        It can also be passed in as a kwarg during invocation.

        You will need to specify a token budget to use this feature. See usage example:

        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            max_tokens=5000,
            thinking={"type": "enabled", "budget_tokens": 2000},
        )

        response = model.invoke("What is the cube root of 50.653?")
        response.content
        ```

        ```python
        [
            {
                "signature": "...",
                "thinking": "To find the cube root of 50.653...",
                "type": "thinking",
            },
            {"text": "The cube root of 50.653 is ...", "type": "text"},
        ]
        ```

        !!! warning "Differences in thinking across model versions"
            The Claude Messages API handles thinking differently across Claude Sonnet
            3.7 and Claude 4 models. Refer to [their docs](https://docs.claude.com/en/docs/build-with-claude/extended-thinking#differences-in-thinking-across-model-versions)
            for more info.

    Citations:
        Anthropic supports a [citations](https://docs.claude.com/en/docs/build-with-claude/citations)
        feature that lets Claude attach context to its answers based on source
        documents supplied by the user. When [document content blocks](https://docs.claude.com/en/docs/build-with-claude/citations#document-types)
        with `"citations": {"enabled": True}` are included in a query, Claude may
        generate citations in its response.

        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-3-5-haiku-20241022")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The grass is green. The sky is blue.",
                        },
                        "title": "My Document",
                        "context": "This is a trustworthy document.",
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "What color is the grass and sky?"},
                ],
            }
        ]
        response = model.invoke(messages)
        response.content
        ```

        ```python
        [
            {"text": "Based on the document, ", "type": "text"},
            {
                "text": "the grass is green",
                "type": "text",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "The grass is green. ",
                        "document_index": 0,
                        "document_title": "My Document",
                        "start_char_index": 0,
                        "end_char_index": 20,
                    }
                ],
            },
            {"text": ", and ", "type": "text"},
            {
                "text": "the sky is blue",
                "type": "text",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "The sky is blue.",
                        "document_index": 0,
                        "document_title": "My Document",
                        "start_char_index": 20,
                        "end_char_index": 36,
                    }
                ],
            },
            {"text": ".", "type": "text"},
        ]
        ```

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 25, "output_tokens": 11, "total_tokens": 36}
        ```

        Message chunks containing token usage will be included during streaming by
        default:

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full.usage_metadata
        ```

        ```python
        {"input_tokens": 25, "output_tokens": 11, "total_tokens": 36}
        ```

        These can be disabled by setting `stream_usage=False` in the stream method,
        or by setting `stream_usage=False` when initializing ChatAnthropic.

    Prompt caching:
        Prompt caching reduces processing time and costs for repetitive tasks or prompts
        with consistent elements

        !!! note
            Only certain models support prompt caching.
            See the [Claude documentation](https://docs.claude.com/en/docs/build-with-claude/prompt-caching#supported-models)
            for a full list.

        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Below is some long context:",
                    },
                    {
                        "type": "text",
                        "text": f"{long_text}",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                "content": "What's that about?",
            },
        ]

        response = model.invoke(messages)
        response.usage_metadata["input_token_details"]
        ```

        ```python
        {"cache_read": 0, "cache_creation": 1458}
        ```

        Alternatively, you may enable prompt caching at invocation time. You may want to
        conditionally cache based on runtime conditions, such as the length of the
        context. Alternatively, this is useful for app-level decisions about what to
        cache.

        ```python
        response = model.invoke(
            messages,
            cache_control={"type": "ephemeral"},
        )
        ```

        ??? note "Extended caching"

            The cache lifetime is 5 minutes by default. If this is too short, you can
            apply one hour caching by setting `ttl` to `'1h'`.

            ```python
            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{long_text}",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        },
                    ],
                }
            ]

            response = model.invoke(messages)
            ```

            Details of cached token counts will be included on the `InputTokenDetails`
            of response's `usage_metadata`:

            ```python
            response = model.invoke(messages)
            response.usage_metadata
            ```

            ```python
            {
                "input_tokens": 1500,
                "output_tokens": 200,
                "total_tokens": 1700,
                "input_token_details": {
                    "cache_read": 0,
                    "cache_creation": 1000,
                    "ephemeral_1h_input_tokens": 750,
                    "ephemeral_5m_input_tokens": 250,
                },
            }
            ```

            See [Claude documentation](https://docs.claude.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration-beta)
            for detail.

    !!! note "Extended context windows (beta)"

        Claude Sonnet 4 supports a 1-million token context window, available in beta for
        organizations in usage tier 4 and organizations with custom rate limits.

        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            betas=["context-1m-2025-08-07"],  # Enable 1M context beta
        )

        long_document = \"\"\"
        This is a very long document that would benefit from the extended 1M
        context window...
        [imagine this continues for hundreds of thousands of tokens]
        \"\"\"

        messages = [
            HumanMessage(f\"\"\"
        Please analyze this document and provide a summary:

        {long_document}

        What are the key themes and main conclusions?
        \"\"\")
        ]

        response = model.invoke(messages)
        ```

        See [Claude documentation](https://docs.claude.com/en/docs/build-with-claude/context-windows#1m-token-context-window)
        for detail.


    !!! note "Token-efficient tool use (beta)"

        See LangChain [docs](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
        for more detail.

        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_core.tools import tool

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
            model_kwargs={
                "extra_headers": {
                    "anthropic-beta": "token-efficient-tools-2025-02-19"
                }
            }
        )

        @tool
        def get_weather(location: str) -> str:
            \"\"\"Get the weather at a location.\"\"\"
            return "It's sunny."

        model_with_tools = model.bind_tools([get_weather])
        response = model_with_tools.invoke(
            "What's the weather in San Francisco?"
        )
        print(response.tool_calls)
        print(f'Total tokens: {response.usage_metadata["total_tokens"]}')
        ```

        ```txt
        [{'name': 'get_weather', 'args': {'location': 'San Francisco'}, 'id': 'toolu_01HLjQMSb1nWmgevQUtEyz17', 'type': 'tool_call'}]
        Total tokens: 408
        ```

    !!! note "Context management"

        Anthropic supports a context editing feature that will automatically manage the
        model's context window (e.g., by clearing tool results).

        See [Anthropic documentation](https://docs.claude.com/en/docs/build-with-claude/context-editing)
        for details and configuration options.

        ```python
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
        )
        model_with_tools = model.bind_tools([{"type": "web_search_20250305", "name": "web_search"}])
        response = model_with_tools.invoke("Search for recent developments in AI")
        ```

    !!! note "Built-in tools"

        See LangChain [docs](https://docs.langchain.com/oss/python/integrations/chat/anthropic#built-in-tools)
        for more detail.

        ??? note "Web search"

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model="claude-3-5-haiku-20241022")

            tool = {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            }
            model_with_tools = model.bind_tools([tool])

            response = model_with_tools.invoke("How do I update a web app to TypeScript 5.5?")
            ```

        ??? note "Web fetch (beta)"

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                betas=["web-fetch-2025-09-10"],  # Enable web fetch beta
            )

            tool = {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "max_uses": 3,
            }
            model_with_tools = model.bind_tools([tool])

            response = model_with_tools.invoke("Please analyze the content at https://example.com/article")
            ```

        ??? note "Code execution"

            ```python
            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                betas=["code-execution-2025-05-22"],
            )

            tool = {"type": "code_execution_20250522", "name": "code_execution"}
            model_with_tools = model.bind_tools([tool])

            response = model_with_tools.invoke(
                "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
            )
            ```

        ??? note "Remote MCP"

            ```python
            from langchain_anthropic import ChatAnthropic

            mcp_servers = [
                {
                    "type": "url",
                    "url": "https://mcp.deepwiki.com/mcp",
                    "name": "deepwiki",
                    "tool_configuration": {  # optional configuration
                        "enabled": True,
                        "allowed_tools": ["ask_question"],
                    },
                    "authorization_token": "PLACEHOLDER",  # optional authorization
                }
            ]

            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                betas=["mcp-client-2025-04-04"],
                mcp_servers=mcp_servers,
            )

            response = model.invoke(
                "What transport protocols does the 2025-03-26 version of the MCP "
                "spec (modelcontextprotocol/modelcontextprotocol) support?"
            )
            ```

        ??? note "Text editor"

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

            tool = {"type": "text_editor_20250124", "name": "str_replace_editor"}
            model_with_tools = model.bind_tools([tool])

            response = model_with_tools.invoke(
                "There's a syntax error in my primes.py file. Can you help me fix it?"
            )
            print(response.text)
            response.tool_calls
            ```

            ```txt
            I'd be happy to help you fix the syntax error in your primes.py file. First, let's look at the current content of the file to identify the error.

            [{'name': 'str_replace_editor',
            'args': {'command': 'view', 'path': '/repo/primes.py'},
            'id': 'toolu_01VdNgt1YV7kGfj9LFLm6HyQ',
            'type': 'tool_call'}]
            ```

        ??? note "Memory tool"

            ```python
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                betas=["context-management-2025-06-27"],
            )
            model_with_tools = model.bind_tools([{"type": "memory_20250818", "name": "memory"}])
            response = model_with_tools.invoke("What are my interests?")
            ```

    !!! note "Response metadata"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "id": "msg_013xU6FHEGEq76aP4RgFerVT",
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 11},
        }
        ```
    """  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
    )

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int | None = Field(default=None, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation."""

    temperature: float | None = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: int | None = None
    """Number of most likely tokens to consider at each step."""

    top_p: float | None = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: float | None = Field(None, alias="timeout")
    """Timeout for requests to Anthropic Completion API."""

    # sdk default = 2: https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#retries
    max_retries: int = 2
    """Number of retries allowed for requests sent to the Anthropic Completion API."""

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
    If neither are set, the default value of `https://api.anthropic.com` will
    be used.
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
    variable."""

    default_headers: Mapping[str, str] | None = None
    """Headers to pass to the Anthropic clients, will be used for every API call."""

    betas: list[str] | None = None
    """List of beta features to enable. If specified, invocations will be routed
    through client.beta.messages.create.

    Example: `betas=["mcp-client-2025-04-04"]`
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output. If `True`, additional
    message chunks will be generated during the stream including usage metadata.
    """

    thinking: dict[str, Any] | None = Field(default=None)
    """Parameters for Claude reasoning,
    e.g., `{"type": "enabled", "budget_tokens": 10_000}`"""

    mcp_servers: list[dict[str, Any]] | None = None
    """List of MCP servers to use for the request.

    Example: `mcp_servers=[{"type": "url", "url": "https://mcp.example.com/mcp",
    "name": "example-mcp"}]`
    """

    context_management: dict[str, Any] | None = None
    """Configuration for
    [context management](https://docs.claude.com/en/docs/build-with-claude/context-editing).
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
        """Set default max_tokens."""
        if values.get("max_tokens") is None:
            model = values.get("model") or values.get("model_name")
            values["max_tokens"] = _default_max_tokens_for(model)
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

        # If cache_control is provided in kwargs, add it to last message
        # and content block.
        if "cache_control" in kwargs and formatted_messages:
            if isinstance(formatted_messages[-1]["content"], list):
                formatted_messages[-1]["content"][-1]["cache_control"] = kwargs.pop(
                    "cache_control"
                )
            elif isinstance(formatted_messages[-1]["content"], str):
                formatted_messages[-1]["content"] = [
                    {
                        "type": "text",
                        "text": formatted_messages[-1]["content"],
                        "cache_control": kwargs.pop("cache_control"),
                    }
                ]
            else:
                pass

        # If cache_control remains in kwargs, it would be passed as a top-level param
        # to the API, but Anthropic expects it nested within a message
        _ = kwargs.pop("cache_control", None)

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

        if "response_format" in payload:
            response_format = payload.pop("response_format")
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                # compat with langchain.agents.create_agent response_format, which is
                # an approximation of OpenAI format
                response_format = cast(dict, response_format["json_schema"]["schema"])
            payload["output_format"] = _convert_to_anthropic_output_format(
                response_format
            )

        if "output_format" in payload and not payload["betas"]:
            payload["betas"] = ["structured-outputs-2025-11-13"]

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
            if (
                isinstance(block, dict)
                and "citations" in block
                and block["citations"] is None
            ):
                block.pop("citations")
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and "text" in block
                and block["text"] is None
            ):
                block.pop("text")

        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
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
            "Anthropic structured output relies on forced tool calling, "
            "which is not supported when `thinking` is enabled. This method will raise "
            "langchain_core.exceptions.OutputParserException if tool calls are not "
            "generated. Consider disabling `thinking` or adjust your prompt to ensure "
            "the tool is called."
        )
        warnings.warn(thinking_admonition, stacklevel=2)
        llm = self.bind_tools(
            [schema],
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
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict[str, str] | str | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        r"""Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports Anthropic format tool schemas and any tool definition handled
                by `langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - name of the tool as a string or as dict `{"type": "tool", "name": "<<tool_name>>"}`: calls corresponding tool;
                - `'auto'`, `{"type: "auto"}`, or `None`: automatically selects a tool (including no tool);
                - `'any'` or `{"type: "any"}`: force at least one tool to be called;
            parallel_tool_calls: Set to `False` to disable parallel tool use.
                Defaults to `None` (no specification, which allows parallel tool use).

                !!! version-added "Added in `langchain-anthropic` 0.3.2"
            strict: If `True`, Claude's schema adherence is applied to tool calls.
                See: [Anthropic docs](https://docs.claude.com/en/docs/build-with-claude/structured-outputs#when-to-use-json-outputs-vs-strict-tool-use).
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

        Example — force tool call with tool_choice `'any'`:

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
            model_with_tools = model.bind_tools([GetWeather, GetPrice], tool_choice="any")
            model_with_tools.invoke(
                "what is the weather like in San Francisco",
            )
            ```

        Example — force specific tool call with `tool_choice` `'<name_of_tool>'`:

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
        model_with_tools = model.bind_tools([GetWeather, GetPrice], tool_choice="GetWeather")
        model_with_tools.invoke("What is the weather like in San Francisco")
        ```

        Example — cache specific tools:

        ```python
        from langchain_anthropic import ChatAnthropic, convert_to_anthropic_tool
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPrice(BaseModel):
            '''Get the price of a specific product.'''

            product: str = Field(..., description="The product to look up.")


        # We'll convert our pydantic class to the anthropic tool format
        # before passing to bind_tools so that we can set the 'cache_control'
        # field on our tool.
        cached_price_tool = convert_to_anthropic_tool(GetPrice)
        # Currently the only supported "cache_control" value is
        # {"type": "ephemeral"}.
        cached_price_tool["cache_control"] = {"type": "ephemeral"}

        # We need to pass in extra headers to enable use of the beta cache
        # control API.
        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
        )
        model_with_tools = model.bind_tools([GetWeather, cached_price_tool])
        model_with_tools.invoke("What is the weather like in San Francisco")
        ```

        This outputs:

        ```python
        AIMessage(
            content=[
                {
                    "text": "Certainly! I can help you find out the current weather in San Francisco. To get this information, I'll use the GetWeather function. Let me fetch that data for you right away.",
                    "type": "text",
                },
                {
                    "id": "toolu_01TS5h8LNo7p5imcG7yRiaUM",
                    "input": {"location": "San Francisco, CA"},
                    "name": "GetWeather",
                    "type": "tool_use",
                },
            ],
            response_metadata={
                "id": "msg_01Xg7Wr5inFWgBxE5jH9rpRo",
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 171,
                    "output_tokens": 96,
                    "cache_creation_input_tokens": 1470,
                    "cache_read_input_tokens": 0,
                },
            },
            id="run-b36a5b54-5d69-470e-a1b0-b932d00b089e-0",
            tool_calls=[
                {
                    "name": "GetWeather",
                    "args": {"location": "San Francisco, CA"},
                    "id": "toolu_01TS5h8LNo7p5imcG7yRiaUM",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 171,
                "output_tokens": 96,
                "total_tokens": 267,
            },
        )
        ```

        If we invoke the tool again, we can see that the "usage" information in the AIMessage.response_metadata shows that we had a cache hit:

        ```python
        AIMessage(
            content=[
                {
                    "text": "To get the current weather in San Francisco, I can use the GetWeather function. Let me check that for you.",
                    "type": "text",
                },
                {
                    "id": "toolu_01HtVtY1qhMFdPprx42qU2eA",
                    "input": {"location": "San Francisco, CA"},
                    "name": "GetWeather",
                    "type": "tool_use",
                },
            ],
            response_metadata={
                "id": "msg_016RfWHrRvW6DAGCdwB6Ac64",
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 171,
                    "output_tokens": 82,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 1470,
                },
            },
            id="run-88b1f825-dcb7-4277-ac27-53df55d22001-0",
            tool_calls=[
                {
                    "name": "GetWeather",
                    "args": {"location": "San Francisco, CA"},
                    "id": "toolu_01HtVtY1qhMFdPprx42qU2eA",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 171,
                "output_tokens": 82,
                "total_tokens": 253,
            },
        )
        ```
        """  # noqa: E501
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
                  [structured output](https://docs.claude.com/en/docs/build-with-claude/structured-outputs)
                  feature.

            kwargs: Additional keyword arguments are ignored.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        Example: Pydantic schema (`include_raw=False`):

        ```python
        from langchain_anthropic import ChatAnthropic
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: str


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

        # -> AnswerWithJustification(
        #     answer='They weigh the same',
        #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
        # )
        ```

        Example: Pydantic schema (`include_raw=True`):

        ```python
        from langchain_anthropic import ChatAnthropic
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: str


        model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
        structured_model = model.with_structured_output(AnswerWithJustification, include_raw=True)

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
        # -> {
        #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
        #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
        #     'parsing_error': None
        # }
        ```

        Example: `dict` schema (`include_raw=False`):

        ```python
        from langchain_anthropic import ChatAnthropic

        schema = {
            "name": "AnswerWithJustification",
            "description": "An answer to the user question along with justification for the answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["answer", "justification"],
            },
        }
        model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
        structured_model = model.with_structured_output(schema)

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
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
            formatted_tool = convert_to_anthropic_tool(schema)
            tool_name = formatted_tool["name"]
            if self.thinking is not None and self.thinking.get("type") == "enabled":
                llm = self._get_llm_for_structured_output_when_thinking_is_enabled(
                    schema,
                    formatted_tool,
                )
            else:
                llm = self.bind_tools(
                    [schema],
                    tool_choice=tool_name,
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
                output_format=_convert_to_anthropic_output_format(schema),
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

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of `dict`, `BaseModel`, function, or `BaseTool`
                objects to be converted to tool schemas.
            kwargs: Additional keyword arguments are passed to the Anthropic
                `messages.count_tokens` method.

        Basic usage:

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

        Pass tool schemas:

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

        !!! warning "Behavior changed in `langchain-anthropic` 0.3.0"

            Uses Anthropic's [token counting API](https://docs.claude.com/en/docs/build-with-claude/token-counting) to count tokens in messages.

        """  # noqa: D214,E501
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
    tool: dict[str, Any] | type | Callable | BaseTool,
    *,
    strict: bool | None = None,
) -> AnthropicTool:
    """Convert a tool-like object to an Anthropic tool definition."""
    # already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
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


class _AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict
    id: str


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


def _convert_to_anthropic_output_format(schema: dict | type) -> dict[str, Any]:
    """Convert JSON schema, Pydantic model, or TypedDict into Claude output_format.

    See: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
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
            text strings. When True, only text content is preserved; when False,
            structured content like tool calls and citations are maintained.
        block_start_event: Previous content block start event, used for tracking
            tool use blocks and maintaining context across related events.

    Returns:
        Tuple containing:
        - AIMessageChunk: Converted message chunk with appropriate content and
          metadata, or None if the event doesn't produce a chunk
        - RawMessageStreamEvent: Updated `block_start_event` for tracking content
          blocks across sequential events, or None if not applicable

    Note:
        Not all Anthropic events result in message chunks. Events like internal
        state changes return None for the message chunk while potentially
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
        content_block["index"] = event.index
        if event.content_block.type == "tool_use":
            tool_call_chunk = create_tool_call_chunk(
                index=event.index,
                id=event.content_block.id,
                name=event.content_block.name,
                args="",
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

    # Process final usage metadata and completion info
    elif event.type == "message_delta" and stream_usage:
        usage_metadata = _create_usage_metadata(event.usage)
        response_metadata = {
            "stop_reason": event.delta.stop_reason,
            "stop_sequence": event.delta.stop_sequence,
        }
        if context_management := getattr(event, "context_management", None):
            response_metadata["context_management"] = context_management.model_dump()
        message_chunk = AIMessageChunk(
            content="" if coerce_content_to_string else [],
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )
        if message_chunk.response_metadata.get("stop_reason"):
            # Mark final Anthropic stream chunk
            message_chunk.chunk_position = "last"
    # Unhandled event types (e.g., `content_block_stop`, `ping` events)
    # https://docs.claude.com/en/docs/build-with-claude/streaming#other-events
    else:
        pass

    if message_chunk:
        message_chunk.response_metadata["model_provider"] = "anthropic"
    return message_chunk, block_start_event


def _create_usage_metadata(anthropic_usage: BaseModel) -> UsageMetadata:
    """Create LangChain `UsageMetadata` from Anthropic `Usage` data.

    Note: Anthropic's `input_tokens` excludes cached tokens, so we manually add
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
