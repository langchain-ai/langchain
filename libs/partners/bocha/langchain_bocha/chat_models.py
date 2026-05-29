"""Bocha chat models."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_bocha._client import DEFAULT_API_BASE, BochaClient


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary in OpenAI format.

    Args:
        message: The message to convert.

    Returns:
        A dictionary representation of the message.
    """
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    msg = f"Got unknown message type: {message}"
    raise TypeError(msg)


class ChatBocha(BaseChatModel):
    """Bocha chat model integration.

    Setup:
        Install `langchain-bocha` and set environment variable `BOCHA_API_KEY`.

        .. code-block:: bash

            pip install -U langchain-bocha
            export BOCHA_API_KEY="your-api-key"

    Key init args — completion params:
        model:
            Name of Bocha model to use, e.g. `'deepseek-v4-pro'`
            or `'deepseek-v4-flash'`.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Bocha API key. If not passed in will be read from env var
            `BOCHA_API_KEY`.

    Instantiate:
        .. code-block:: python

            from langchain_bocha import ChatBocha

            model = ChatBocha(
                model="deepseek-v4-pro",
                temperature=0.7,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Explain DSA sparse attention in one sentence."),
            ]
            model.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in model.stream(messages):
                print(chunk.content, end="", flush=True)
    """

    client: Any = Field(default=None, exclude=True)
    """The underlying `BochaClient` instance."""

    model: str = "deepseek-v4-pro"
    """The name of the model."""

    bocha_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("BOCHA_API_KEY", default=None),
        alias="api_key",
    )
    """Bocha API key."""

    api_base: str = Field(default=DEFAULT_API_BASE, alias="base_url")
    """Bocha API base URL."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    request_timeout: float | None = Field(None, alias="timeout")
    """Timeout for requests to Bocha completion API."""

    max_retries: int = 2
    """Maximum number of retries to make when generating."""

    streaming: bool = False
    """Whether to stream the results or not."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-bocha"

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the API key is present and initialize the client."""
        api_key = self.bocha_api_key.get_secret_value() if self.bocha_api_key else ""
        if not api_key:
            msg = "BOCHA_API_KEY must be set."
            raise ValueError(msg)

        if not self.client:
            self.client = BochaClient(
                api_key=api_key,
                base_url=self.api_base,
                timeout=self.request_timeout or 60,
                max_retries=self.max_retries,
            )
        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling the Bocha chat API."""
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return {**params, **self.model_kwargs}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate output from a list of messages.

        Args:
            messages: The list of messages to generate from.
            stop: A list of stop sequences.
            run_manager: The callback manager.
            kwargs: Additional parameters.

        Returns:
            The generated ChatResult.
        """
        if self.streaming:
            return self._generate_from_stream(messages, stop, run_manager, **kwargs)

        payload = self._build_payload(messages, stop=stop, **kwargs)
        res_json = self.client.post("/chat/completions", payload)
        return self._parse_response(res_json)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate output from a list of messages asynchronously.

        Args:
            messages: The list of messages to generate from.
            stop: A list of stop sequences.
            run_manager: The callback manager.
            kwargs: Additional parameters.

        Returns:
            The generated ChatResult.
        """
        payload = self._build_payload(messages, stop=stop, **kwargs)
        res_json = await self.client.apost("/chat/completions", payload)
        return self._parse_response(res_json)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the generated output.

        Args:
            messages: The list of messages to generate from.
            stop: A list of stop sequences.
            run_manager: The callback manager.
            kwargs: Additional parameters.

        Yields:
            The ChatGenerationChunks.
        """
        payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)
        for parsed in self.client.post_stream("/chat/completions", payload):
            choice = parsed.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if content:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        response_metadata=parsed,
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the generated output asynchronously.

        Args:
            messages: The list of messages to generate from.
            stop: A list of stop sequences.
            run_manager: The callback manager.
            kwargs: Additional parameters.

        Yields:
            The ChatGenerationChunks.
        """
        payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)
        async for parsed in self.client.apost_stream("/chat/completions", payload):
            choice = parsed.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if content:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        response_metadata=parsed,
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    # -- private helpers --------------------------------------------------

    def _build_payload(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request payload."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        payload: dict[str, Any] = {
            **self._default_params,
            "messages": message_dicts,
        }
        if stop:
            payload["stop"] = stop
        if stream:
            payload["stream"] = True
        payload.update(kwargs)
        return payload

    def _generate_from_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Collect streaming chunks into a single ChatResult."""
        full_content = ""
        last_metadata: dict[str, Any] = {}
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            full_content += str(chunk.message.content)
            if chunk.message.response_metadata:
                last_metadata = chunk.message.response_metadata
        ai_message = AIMessage(
            content=full_content,
            response_metadata=last_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    @staticmethod
    def _parse_response(res_json: dict[str, Any]) -> ChatResult:
        """Parse a non-streaming JSON response into a ChatResult."""
        choice = res_json.get("choices", [{}])[0]
        message_data = choice.get("message", {})
        content = message_data.get("content", "")

        usage = res_json.get("usage", {})
        generation_info: dict[str, Any] = {}
        if "finish_reason" in choice:
            generation_info["finish_reason"] = choice["finish_reason"]

        ai_message = AIMessage(
            content=content,
            response_metadata=res_json,
            usage_metadata={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            if usage
            else None,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=ai_message, generation_info=generation_info)
            ]
        )
