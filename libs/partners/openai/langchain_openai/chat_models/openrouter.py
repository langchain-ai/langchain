"""OpenRouter chat wrapper with Anthropic prompt caching support.

This module provides a ChatOpenRouter class that integrates with OpenRouter's API
while preserving Anthropic's prompt caching features via cache_control fields.

The OpenAI Python SDK validates message schemas and filters out non-standard fields
like cache_control. This implementation uses httpx directly to bypass SDK validation,
ensuring cache_control fields are properly transmitted to OpenRouter.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from typing import (
    Any,
)

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert LangChain message to OpenRouter format while preserving cache_control.

    Args:
        message: A LangChain message object.

    Returns:
        Dictionary representation compatible with OpenRouter API, including any
        cache_control fields for prompt caching.
    """
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = "user"

    content = message.content
    result: dict[str, Any] = {"role": role}

    # Preserve content structure for cache_control
    if isinstance(content, str):
        result["content"] = content
    elif isinstance(content, list):
        # Preserve list format including cache_control blocks
        result["content"] = content
    else:
        result["content"] = str(content)

    # Handle tool_calls for AIMessage
    if isinstance(message, AIMessage) and message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": (
                        json.dumps(tc.get("args", {}))
                        if isinstance(tc.get("args"), dict)
                        else tc.get("args", "{}")
                    ),
                },
            }
            for tc in message.tool_calls
        ]
        # Content can be null when tool_calls are present
        if not result["content"]:
            result["content"] = None

    # Handle tool_call_id for ToolMessage
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id

    return result


def _convert_tool_to_dict(tool: Any) -> dict[str, Any]:
    """Convert tool to OpenRouter format while preserving cache_control.

    Args:
        tool: A tool object (dict or LangChain tool).

    Returns:
        Dictionary representation compatible with OpenRouter API.
    """
    if isinstance(tool, dict):
        return tool

    # Handle LangChain tool objects
    if hasattr(tool, "name") and hasattr(tool, "description"):
        tool_dict: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
            },
        }
        if hasattr(tool, "args_schema") and tool.args_schema:
            tool_dict["function"]["parameters"] = tool.args_schema.schema()

        # Preserve cache_control if present
        if hasattr(tool, "cache_control"):
            tool_dict["cache_control"] = tool.cache_control

        return tool_dict

    return tool


class ChatOpenRouter(BaseChatModel):
    """OpenRouter chat model integration with Anthropic prompt caching support.

    This implementation uses httpx directly to bypass OpenAI SDK schema validation,
    ensuring cache_control fields are properly transmitted to OpenRouter for
    Anthropic prompt caching support.

    Setup:
        Install ``langchain-openai`` and set environment variable
        ``OPENROUTER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-openai
            export OPENROUTER_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of OpenRouter model to use.
        temperature: float
            Sampling temperature.
        max_tokens: int
            Max number of tokens to generate.

    Key init args — client params:
        timeout: int
            Timeout for requests in seconds.
        max_retries: int
            Max number of retries for failed requests.
        api_key: SecretStr | None
            OpenRouter API key.
        base_url: str | None
            Base URL for API requests.

    Instantiate:
        .. code-block:: python

            from langchain_openai import ChatOpenRouter

            llm = ChatOpenRouter(
                model="anthropic/claude-3.5-sonnet",
                temperature=0,
                max_tokens=1024,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator."),
                ("human", "Translate 'hello' to French."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content="Bonjour")

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    With prompt caching (Anthropic models):
        .. code-block:: python

            from langchain_core.messages import SystemMessage

            system_msg = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are a helpful assistant...",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
            llm.invoke([system_msg, ("human", "Hello")])
    """

    model: str = Field(default="google/gemini-flash-1.5")
    """Model name to use."""

    api_key: SecretStr | None = Field(default=None, alias="openai_api_key")
    """OpenRouter API key."""

    base_url: str | None = Field(default=None, alias="openai_api_base")
    """Base URL for OpenRouter API."""

    max_tokens: int = Field(default=4096)
    """Maximum number of tokens to generate."""

    temperature: float = Field(default=0.7)
    """Sampling temperature."""

    timeout: int = Field(default=120)
    """Request timeout in seconds."""

    max_retries: int = Field(default=3)
    """Maximum number of retries for failed requests."""

    default_headers: dict[str, str] | None = Field(default=None)
    """Optional default headers to include in requests."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openrouter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _get_api_base(self) -> str:
        """Get API base URL."""
        return self.base_url or os.getenv(
            "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
        )

    def _get_api_key(self) -> str:
        """Get API key from instance or environment."""
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("OPENROUTER_API_KEY", "")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response.

        Args:
            messages: List of messages to send to the model.
            stop: List of stop sequences.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Returns:
            ChatResult containing the model's response.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
            RuntimeError: If all retry attempts fail.
        """
        # Convert messages, preserving cache_control
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": message_dicts,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add tools if provided
        if kwargs.get("tools"):
            payload["tools"] = [_convert_tool_to_dict(t) for t in kwargs["tools"]]

        # Add stop sequences
        if stop:
            payload["stop"] = stop

        # Request usage statistics
        payload["usage"] = {"include": True}

        # Prepare request
        api_base = self._get_api_base().rstrip("/")
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        if self.default_headers:
            headers.update(self.default_headers)

        # Retry logic with exponential backoff
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                # Parse response
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")

                # Parse tool_calls
                tool_calls = []
                if "tool_calls" in message:
                    tool_calls.extend(
                        {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "args": json.loads(
                                tc.get("function", {}).get("arguments", "{}")
                            ),
                        }
                        for tc in message["tool_calls"]
                    )

                # Create AIMessage
                ai_message = AIMessage(
                    content=content or "",
                    tool_calls=tool_calls if tool_calls else [],
                )

                # Parse usage statistics
                usage = data.get("usage", {})
                generation_info: dict[str, Any] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

                # Parse cache statistics
                prompt_details = usage.get("prompt_tokens_details", {})
                cached_tokens = prompt_details.get("cached_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                if cached_tokens and prompt_tokens:
                    cache_hit_rate = cached_tokens / prompt_tokens * 100
                    logger.info(
                        "Cache hit=%s/%s (%.1f%%)",
                        cached_tokens,
                        prompt_tokens,
                        cache_hit_rate,
                    )
                    generation_info["cached_tokens"] = cached_tokens
                    generation_info["cache_hit_rate"] = cache_hit_rate

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=ai_message, generation_info=generation_info
                        )
                    ],
                    llm_output={
                        "model": data.get("model", self.model),
                        "token_usage": generation_info,
                    },
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = 2**attempt
                    logger.warning("Rate limited, waiting %ss before retry", wait_time)
                    time.sleep(wait_time)
                    continue
                if e.response.status_code >= 500:
                    # Server error - retry
                    time.sleep(1)
                    continue
                # Other errors - raise immediately
                logger.error(
                    "HTTP error %s: %s", e.response.status_code, e.response.text
                )
                raise
            except Exception as e:
                last_error = e
                logger.error("Error on attempt %s: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        if last_error:
            raise last_error
        msg = "Failed to get response from OpenRouter"
        raise RuntimeError(msg)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response.

        Args:
            messages: List of messages to send to the model.
            stop: List of stop sequences.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments.

        Yields:
            ChatGenerationChunk for each token in the response.
        """
        # Convert messages
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": message_dicts,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if kwargs.get("tools"):
            payload["tools"] = [_convert_tool_to_dict(t) for t in kwargs["tools"]]

        if stop:
            payload["stop"] = stop

        api_base = self._get_api_base().rstrip("/")
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        if self.default_headers:
            headers.update(self.default_headers)

        with (
            httpx.Client(timeout=self.timeout) as client,
            client.stream("POST", url, json=payload, headers=headers) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=content)
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(content)
                            yield chunk
                    except json.JSONDecodeError:
                        continue
