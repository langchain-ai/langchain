from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class ChatSarvam(BaseChatModel):
    """
    `client` and `async_client` are instances of sarvamai.SarvamAI and sarvamai.AsyncSarvamAI
    respectively. They are not serialized when the model is saved or loaded.
    """
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(alias="model")
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int = 1
    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")

    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    seed: int | None = None
    wiki_grounding: bool | None = None

    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    sarvam_api_key: SecretStr | None = Field(
        alias="api_key", default_factory=secret_from_env("SARVAM_API_KEY", default=None)
    )
    request_timeout: float | None = Field(default=None, alias="timeout")

    http_client: Any | None = None
    http_async_client: Any | None = None

    streaming: bool = False

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                extra[field_name] = values.pop(field_name)
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                "Instead they were passed in as part of `model_kwargs` parameter."
            )
        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        try:
            from sarvamai import AsyncSarvamAI, SarvamAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import sarvamai python package. Please install it with `pip install sarvamai`."
            ) from exc

        client_params: dict[str, Any] = {
            "api_subscription_key": (
                self.sarvam_api_key.get_secret_value() if self.sarvam_api_key else None
            ),
            "timeout": self.request_timeout,
        }

        if client_params["api_subscription_key"] is None:
            raise ValueError(
                "Sarvam API key is not set. Set `sarvam_api_key` field or `SARVAM_API_KEY` env var."
            )

        if not self.client:
            sync_specific: dict[str, Any] = {}
            if self.http_client is not None:
                sync_specific["httpx_client"] = self.http_client
            self.client = SarvamAI(**client_params, **sync_specific).chat
        if not self.async_client:
            async_specific: dict[str, Any] = {}
            if self.http_async_client is not None:
                async_specific["httpx_client"] = self.http_async_client
            self.async_client = AsyncSarvamAI(
                **client_params, **async_specific
            ).chat
        return self

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"sarvam_api_key": "SARVAM_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _llm_type(self) -> str:
        return "sarvam-chat"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        # Do not forward params to API here; only produce tracing metadata.
        ls_params = LangSmithParams(
            ls_provider="sarvam",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=self.temperature,
        )
        if isinstance(self.max_tokens, int):
            ls_params["ls_max_tokens"] = self.max_tokens
        if stop or self.stop:
            ls_stop: list[str] | None
            if stop is not None:
                ls_stop = stop
            elif isinstance(self.stop, list):
                ls_stop = self.stop
            elif isinstance(self.stop, str):
                ls_stop = [self.stop]
            else:
                ls_stop = None
            if ls_stop:
                ls_params["ls_stop"] = ls_stop
        return ls_params

    def _default_params(self) -> dict[str, Any]:  # type: ignore[override]
        # Sarvam SDK does not accept a 'model' parameter currently; default model is sarvam-m.
        params: dict[str, Any] = {
            "n": self.n,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.stop is not None:
            params["stop"] = self.stop
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            params["seed"] = self.seed
        if self.wiki_grounding is not None:
            params["wiki_grounding"] = self.wiki_grounding
        if self.model_kwargs:
            params.update(self.model_kwargs)
        return params

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params()
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        resp = self.client.completions(messages=message_dicts, **params)
        return self._create_chat_result(resp, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        resp = await self.async_client.completions(messages=message_dicts, **params)
        return self._create_chat_result(resp, params)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.completions(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # type: ignore[attr-defined]
            if len(chunk.get("choices", [])) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in self.async_client.completions(
            messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # type: ignore[attr-defined]
            if len(chunk.get("choices", [])) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                if model_name := chunk.get("model"):
                    generation_info["model_name"] = model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _create_chat_result(self, response: dict | BaseModel, params: Mapping[str, Any]) -> ChatResult:
        generations: list[ChatGeneration] = []
        if not isinstance(response, dict):
            response = response.model_dump()  # type: ignore[attr-defined]
        token_usage = response.get("usage", {})
        for res in response.get("choices", []):
            message = _convert_dict_to_message(res["message"])  # type: ignore[index]
            if token_usage and isinstance(message, AIMessage):
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                message.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": token_usage.get(
                        "total_tokens", input_tokens + output_tokens
                    ),
                }
            generation_info: dict[str, Any] = {"finish_reason": res.get("finish_reason")}
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output: dict[str, Any] = {}
        if token_usage:
            llm_output["token_usage"] = token_usage
        if model_name := response.get("model"):
            llm_output["model_name"] = model_name
        if system_fingerprint := response.get("system_fingerprint"):
            llm_output["system_fingerprint"] = system_fingerprint
        return ChatResult(generations=generations, llm_output=llm_output or None)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, list):
            text_blocks = [
                block for block in content if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = text_blocks if text_blocks else ""
        return {"role": "assistant", "content": content}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, FunctionMessage):
        return {"role": "function", "content": message.content, "name": message.name}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    raise TypeError(f"Got unknown type {message}")


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    delta = cast("Mapping[str, Any]", choice.get("delta", {}))
    role = cast("str | None", delta.get("role"))
    content = cast("str", delta.get("content") or "")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=delta.get("name"))  # type: ignore[arg-type]
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=delta.get("tool_call_id"))  # type: ignore[arg-type]
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        return AIMessage(
            content=_dict.get("content", "") or "",
            response_metadata={"model_provider": "sarvam"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    if role == "tool":
        return ToolMessage(content=_dict.get("content", ""), tool_call_id=_dict.get("tool_call_id"))  # type: ignore[arg-type]
    return ChatMessage(content=_dict.get("content", ""), role=cast("str", role))
