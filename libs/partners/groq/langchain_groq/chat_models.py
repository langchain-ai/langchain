"""Groq Chat wrapper."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
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
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_groq._compat import _convert_from_v1_to_groq
from langchain_groq.version import __version__


class ChatGroq(BaseChatModel):
    r"""Groq Chat large language models API.

    To use, you should have the
    environment variable `GROQ_API_KEY` set with your API key.

    Any parameters that are valid to be passed to the groq.create call
    can be passed in, even if not explicitly saved on this class.

    Setup:
        Install `langchain-groq` and set environment variable
        `GROQ_API_KEY`.

        ```bash
        pip install -U langchain-groq
        export GROQ_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Groq model to use, e.g. `llama-3.1-8b-instant`.
        temperature:
            Sampling temperature. Ranges from `0.0` to `1.0`.
        max_tokens:
            Max number of tokens to generate.
        reasoning_format:
            The format for reasoning output. Groq will default to `raw` if left
            undefined.

            - `'parsed'`: Separates reasoning into a dedicated field while keeping the
                response concise. Reasoning will be returned in the
                `additional_kwargs.reasoning_content` field of the response.
            - `'raw'`: Includes reasoning within think tags (e.g.
                `<think>{reasoning_content}</think>`).
            - `'hidden'`: Returns only the final answer content. Note: this only
                suppresses reasoning content in the response; the model will still perform
                reasoning unless overridden in `reasoning_effort`.

            See the [Groq documentation](https://console.groq.com/docs/reasoning#reasoning)
            for more details and a list of supported models.
        model_kwargs:
            Holds any model parameters valid for create call not
            explicitly specified.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Groq API key. If not passed in will be read from env var `GROQ_API_KEY`.
        base_url:
            Base URL path for API requests, leave blank if not using a proxy
            or service emulator.
        custom_get_token_ids:
            Optional encoder to use for counting tokens.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        ```python
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_retries=2,
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```
        ```python
        AIMessage(content='The English sentence "I love programming" can
        be translated to French as "J\'aime programmer". The word
        "programming" is translated as "programmer" in French.',
        response_metadata={'token_usage': {'completion_tokens': 38,
        'prompt_tokens': 28, 'total_tokens': 66, 'completion_time':
        0.057975474, 'prompt_time': 0.005366091, 'queue_time': None,
        'total_time': 0.063341565}, 'model_name': 'llama-3.1-8b-instant',
        'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop',
        'logprobs': None}, id='run-ecc71d70-e10c-4b69-8b8c-b8027d95d4b8-0')
        ```

    Stream:
        ```python
        # Streaming `text` for each content chunk received
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```

        ```python
        content='' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        content='The' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        content=' English' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        content=' sentence' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        ...
        content=' program' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        content='".' id='run-4e9f926b-73f5-483b-8ef5-09533d925853'
        content='' response_metadata={'finish_reason': 'stop'}
        id='run-4e9f926b-73f5-483b-8ef5-09533d925853
        ```

        ```python
        # Reconstructing a full response
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(content='The English sentence "I love programming"
        can be translated to French as "J\'aime programmer". Here\'s the
        breakdown of the sentence: "J\'aime" is the French equivalent of "
        I love", and "programmer" is the French infinitive for "to program".
        So, the literal translation is "I love to program". However, in
        English we often omit the "to" when talking about activities we
        love, and the same applies to French. Therefore, "J\'aime
        programmer" is the correct and natural way to express "I love
        programming" in French.', response_metadata={'finish_reason':
        'stop'}, id='run-a3c35ac4-0750-4d08-ac55-bfc63805de76')
        ```

    Async:
        ```python
        await model.ainvoke(messages)
        ```

        ```python
        AIMessage(content='The English sentence "I love programming" can
        be translated to French as "J\'aime programmer". The word
        "programming" is translated as "programmer" in French. I hope
        this helps! Let me know if you have any other questions.',
        response_metadata={'token_usage': {'completion_tokens': 53,
        'prompt_tokens': 28, 'total_tokens': 81, 'completion_time':
        0.083623752, 'prompt_time': 0.007365126, 'queue_time': None,
        'total_time': 0.090988878}, 'model_name': 'llama-3.1-8b-instant',
        'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop',
        'logprobs': None}, id='run-897f3391-1bea-42e2-82e0-686e2367bcf8-0')
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
        ai_msg = model_with_tools.invoke("What is the population of NY?")
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetPopulation",
                "args": {"location": "NY"},
                "id": "call_bb8d",
            }
        ]
        ```

        See `ChatGroq.bind_tools()` method for more.

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
            setup="Why don't cats play poker in the jungle?",
            punchline="Too many cheetahs!",
            rating=None,
        )
        ```

        See `ChatGroq.with_structured_output()` for more.

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "token_usage": {
                "completion_tokens": 70,
                "prompt_tokens": 28,
                "total_tokens": 98,
                "completion_time": 0.111956391,
                "prompt_time": 0.007518279,
                "queue_time": None,
                "total_time": 0.11947467,
            },
            "model_name": "llama-3.1-8b-instant",
            "system_fingerprint": "fp_c5f20b5bb1",
            "finish_reason": "stop",
            "logprobs": None,
        }
        ```
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)

    async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(alias="model")
    """Model name to use."""

    temperature: float = 0.7
    """What sampling temperature to use."""

    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    reasoning_format: Literal["parsed", "raw", "hidden"] | None = Field(default=None)
    """The format for reasoning output. Groq will default to raw if left undefined.

    - `'parsed'`: Separates reasoning into a dedicated field while keeping the
        response concise. Reasoning will be returned in the
        `additional_kwargs.reasoning_content` field of the response.
    - `'raw'`: Includes reasoning within think tags (e.g.
        `<think>{reasoning_content}</think>`).
    - `'hidden'`: Returns only the final answer content. Note: this only suppresses
        reasoning content in the response; the model will still perform reasoning unless
        overridden in `reasoning_effort`.

    See the [Groq documentation](https://console.groq.com/docs/reasoning#reasoning)
    for more details and a list of supported models.
    """

    reasoning_effort: str | None = Field(default=None)
    """The level of effort the model will put into reasoning. Groq will default to
    enabling reasoning if left undefined.

    See the [Groq documentation](https://console.groq.com/docs/reasoning#options-for-reasoning-effort)
    for more details and a list of options and models that support setting a reasoning
    effort.
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    groq_api_key: SecretStr | None = Field(
        alias="api_key", default_factory=secret_from_env("GROQ_API_KEY", default=None)
    )
    """Automatically inferred from env var `GROQ_API_KEY` if not provided."""

    groq_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("GROQ_API_BASE", default=None)
    )
    """Base URL path for API requests. Leave blank if not using a proxy or service
    emulator.
    """

    # to support explicit proxy for Groq
    groq_proxy: str | None = Field(default_factory=from_env("GROQ_PROXY", default=None))

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Groq completion API. Can be float, `httpx.Timeout` or
    `None`.
    """

    max_retries: int = 2
    """Maximum number of retries to make when generating."""

    streaming: bool = False
    """Whether to stream the results or not."""

    n: int = 1
    """Number of chat completions to generate for each prompt."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    service_tier: Literal["on_demand", "flex", "auto"] = Field(default="on_demand")
    """Optional parameter that you can include to specify the service tier you'd like to
    use for requests.

    - `'on_demand'`: Default.
    - `'flex'`: On-demand processing when capacity is available, with rapid timeouts
        if resources are constrained. Provides balance between performance and
        reliability for workloads that don't require guaranteed processing.
    - `'auto'`: Uses on-demand rate limits, then falls back to `'flex'` if those
        limits are exceeded

    See the [Groq documentation](https://console.groq.com/docs/flex-processing) for more
    details and a list of service tiers and descriptions.
    """

    default_headers: Mapping[str, str] | None = None

    default_query: Mapping[str, object] | None = None

    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Any | None = None
    """Optional `httpx.Client`."""

    http_async_client: Any | None = None
    """Optional `httpx.AsyncClient`. Only used for async invocations. Must specify
        `http_client` as well if you'd like a custom client for sync invocations."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended.""",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)
        if self.temperature == 0:
            self.temperature = 1e-8

        default_headers = {"User-Agent": f"langchain/{__version__}"} | dict(
            self.default_headers or {}
        )

        client_params: dict[str, Any] = {
            "api_key": (
                self.groq_api_key.get_secret_value() if self.groq_api_key else None
            ),
            "base_url": self.groq_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": default_headers,
            "default_query": self.default_query,
        }

        try:
            import groq  # noqa: PLC0415

            sync_specific: dict[str, Any] = {"http_client": self.http_client}
            if not self.client:
                self.client = groq.Groq(
                    **client_params, **sync_specific
                ).chat.completions
            if not self.async_client:
                async_specific: dict[str, Any] = {"http_client": self.http_async_client}
                self.async_client = groq.AsyncGroq(
                    **client_params, **async_specific
                ).chat.completions
        except ImportError as exc:
            msg = (
                "Could not import groq python package. "
                "Please install it with `pip install groq`."
            )
            raise ImportError(msg) from exc
        return self

    #
    # Serializable class method overrides
    #
    @property
    def lc_secrets(self) -> dict[str, str]:
        """Mapping of secret environment variables."""
        return {"groq_api_key": "GROQ_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    #
    # BaseChatModel method overrides
    #
    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "groq-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="groq",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: CallbackManagerForLLMRun
        | AsyncCallbackManagerForLLMRun
        | None = None,
        **kwargs: Any,
    ) -> bool:
        """Determine if a given model call should hit the streaming API."""
        base_should_stream = super()._should_stream(
            async_api=async_api, run_manager=run_manager, **kwargs
        )
        if base_should_stream and ("response_format" in kwargs):
            # Streaming not supported in JSON mode or structured outputs.
            response_format = kwargs["response_format"]
            if isinstance(response_format, dict) and response_format.get("type") in {
                "json_schema",
                "json_object",
            }:
                return False
        return base_should_stream

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response, params)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
        }
        response = await self.async_client.create(messages=message_dicts, **params)
        return self._create_chat_result(response, params)

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
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # noqa: PLW2901
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                service_tier = params.get("service_tier") or self.service_tier
                generation_info["service_tier"] = service_tier
                reasoning_effort = (
                    params.get("reasoning_effort") or self.reasoning_effort
                )
                if reasoning_effort:
                    generation_info["reasoning_effort"] = reasoning_effort
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )

            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
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
        async for chunk in await self.async_client.create(
            messages=message_dicts, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()  # noqa: PLW2901
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
                if system_fingerprint := chunk.get("system_fingerprint"):
                    generation_info["system_fingerprint"] = system_fingerprint
                service_tier = params.get("service_tier") or self.service_tier
                generation_info["service_tier"] = service_tier
                reasoning_effort = (
                    params.get("reasoning_effort") or self.reasoning_effort
                )
                if reasoning_effort:
                    generation_info["reasoning_effort"] = reasoning_effort
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs

            if generation_info:
                message_chunk = message_chunk.model_copy(
                    update={"response_metadata": generation_info}
                )

            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info or None
            )

            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    #
    # Internal methods
    #
    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling Groq API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "stop": self.stop,
            "reasoning_format": self.reasoning_format,
            "reasoning_effort": self.reasoning_effort,
            "service_tier": self.service_tier,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def _create_chat_result(
        self, response: dict | BaseModel, params: dict
    ) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        llm_output["service_tier"] = params.get("service_tier") or self.service_tier
        reasoning_effort = params.get("reasoning_effort") or self.reasoning_effort
        if reasoning_effort:
            llm_output["reasoning_effort"] = reasoning_effort
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage and v is not None:
                        # Handle nested dictionaries
                        if isinstance(v, dict):
                            if k not in overall_token_usage:
                                overall_token_usage[k] = {}
                            for nested_k, nested_v in v.items():
                                if (
                                    nested_k in overall_token_usage[k]
                                    and nested_v is not None
                                ):
                                    overall_token_usage[k][nested_k] += nested_v
                                else:
                                    overall_token_usage[k][nested_k] = nested_v
                        else:
                            overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        if self.service_tier:
            combined["service_tier"] = self.service_tier
        return combined

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                `langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                `'auto'` to automatically determine which function to call
                with the option to not call any function, `'any'` to enforce that some
                function is called, or a dict of the form:
                `{"type": "function", "function": {"name": <<tool_name>>}}`.
            **kwargs: Any additional parameters to pass to the
                `langchain.runnable.Runnable` constructor.

        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    msg = (
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                    raise ValueError(msg)
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | None = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

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

                !!! warning "Behavior changed in `langchain-groq` 0.3.8"
                    Added support for Groq's dedicated structured output feature via
                    `method="json_schema"`.

            method: The method for steering model generation, one of:

                - `'function_calling'`:
                    Uses Groq's tool-calling [API](https://console.groq.com/docs/tool-use)
                - `'json_schema'`:
                    Uses Groq's [Structured Output API](https://console.groq.com/docs/structured-outputs).
                    Supported for a subset of models, including `openai/gpt-oss`,
                    `moonshotai/kimi-k2-instruct-0905`, and some `meta-llama/llama-4`
                    models. See [docs](https://console.groq.com/docs/structured-outputs)
                    for details.
                - `'json_mode'`:
                    Uses Groq's [JSON mode](https://console.groq.com/docs/structured-outputs#json-object-mode).
                    Note that if using JSON mode then you must include instructions for
                    formatting the output into the desired schema into the model call

                Learn more about the differences between the methods and which models
                support which methods [here](https://console.groq.com/docs/structured-outputs).

            method:
                The method for steering model generation, either `'function_calling'`
                or `'json_mode'`. If `'function_calling'` then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If `'json_mode'` then JSON mode will be used.

                !!! note
                    If using `'json_mode'` then you must include instructions for formatting
                    the output into the desired schema into the model call. (either via the
                    prompt itself or in the system message/prompt/instructions).

                !!! warning
                    `'json_mode'` does not support streaming responses stop sequences.

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            kwargs:
                Any additional parameters to pass to the `langchain.runnable.Runnable`
                constructor.

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

        Example: schema=Pydantic class, method="function_calling", include_raw=False:

        ```python
        from typing import Optional

        from langchain_groq import ChatGroq
        from pydantic import BaseModel, Field


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            # If we provide default values and/or descriptions for fields, these will be passed
            # to the model. This is an important part of improving a model's ability to
            # correctly return structured outputs.
            justification: str | None = Field(default=None, description="A justification for the answer.")


        model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

        # -> AnswerWithJustification(
        #     answer='They weigh the same',
        #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
        # )
        ```

        Example: schema=Pydantic class, method="function_calling", include_raw=True:

        ```python
        from langchain_groq import ChatGroq
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: str


        model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_model = model.with_structured_output(
            AnswerWithJustification,
            include_raw=True,
        )

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
        # -> {
        #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
        #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
        #     'parsing_error': None
        # }
        ```

        Example: schema=TypedDict class, method="function_calling", include_raw=False:

        ```python
        from typing_extensions import Annotated, TypedDict

        from langchain_groq import ChatGroq


        class AnswerWithJustification(TypedDict):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            justification: Annotated[str | None, None, "A justification for the answer."]


        model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_model = model.with_structured_output(AnswerWithJustification)

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")
        # -> {
        #     'answer': 'They weigh the same',
        #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
        # }
        ```

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:

        ```python
        from langchain_groq import ChatGroq

        oai_schema = {
            'name': 'AnswerWithJustification',
            'description': 'An answer to the user question along with justification for the answer.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {'type': 'string'},
                    'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                },
                'required': ['answer']
            }

            model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
            structured_model = model.with_structured_output(oai_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'answer': 'They weigh the same',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            # }
        ```

        Example: schema=Pydantic class, method="json_schema", include_raw=False:

        ```python
        from typing import Optional

        from langchain_groq import ChatGroq
        from pydantic import BaseModel, Field


        class AnswerWithJustification(BaseModel):
            '''An answer to the user question along with justification for the answer.'''

            answer: str
            # If we provide default values and/or descriptions for fields, these will be passed
            # to the model. This is an important part of improving a model's ability to
            # correctly return structured outputs.
            justification: str | None = Field(default=None, description="A justification for the answer.")


        model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_model = model.with_structured_output(
            AnswerWithJustification,
            method="json_schema",
        )

        structured_model.invoke("What weighs more a pound of bricks or a pound of feathers")

        # -> AnswerWithJustification(
        #     answer='They weigh the same',
        #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
        # )
        ```

        Example: schema=Pydantic class, method="json_mode", include_raw=True:

        ```python
        from langchain_groq import ChatGroq
        from pydantic import BaseModel


        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str


        model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_model = model.with_structured_output(
            AnswerWithJustification, method="json_mode", include_raw=True
        )

        structured_model.invoke(
            "Answer the following question. "
            "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
            "What's heavier a pound of bricks or a pound of feathers?"
        )
        # -> {
        #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
        #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
        #     'parsing_error': None
        # }
        ```

        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = (
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            # Use structured outputs (json_schema) for models that support it
            # Convert schema to JSON Schema format for structured outputs
            if schema is None:
                msg = (
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
                raise ValueError(msg)
            json_schema = convert_to_json_schema(schema)
            schema_name = json_schema.get("title", "")
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }
            ls_format_info = {
                "kwargs": {"method": "json_schema"},
                "schema": json_schema,
            }
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format=ls_format_info,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )

        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


#
# Type conversion helpers
#
def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.

    """
    message_dict: dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        # Translate v1 content
        if message.response_metadata.get("output_version") == "v1":
            new_content, new_additional_kwargs = _convert_from_v1_to_groq(
                message.content_blocks, message.response_metadata.get("model_provider")
            )
            message = message.model_copy(
                update={
                    "content": new_content,
                    "additional_kwargs": new_additional_kwargs,
                }
            )
        message_dict = {"role": "assistant", "content": message.content}

        # If content is a list of content blocks, filter out tool_call blocks
        # as Groq API only accepts 'text' type blocks in content
        if isinstance(message.content, list):
            text_blocks = [
                block
                for block in message.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            message_dict["content"] = text_blocks if text_blocks else ""

        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_groq_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_groq_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
            # If tool calls only (no text blocks), content is None not empty string
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "" or (
                isinstance(message_dict["content"], list)
                and not message_dict["content"]
            ):
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown type {message}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        # Groq sends 'null' (JSON null) for tools with no arguments, but we
        # expect '{}' (empty JSON object) to represent empty arguments
        tool_calls = _dict["tool_calls"]
        for tool_call in tool_calls:
            if (
                tool_call.get("function")
                and tool_call["function"].get("arguments") == "null"
            ):
                tool_call["function"]["arguments"] = "{}"
        additional_kwargs["tool_calls"] = tool_calls

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if executed_tools := _dict.get("executed_tools"):
            additional_kwargs["executed_tools"] = []
            for executed_tool in executed_tools:
                if executed_tool.get("output"):
                    # Tool output duplicates query and other server tool call data
                    additional_kwargs["executed_tools"].append(
                        {
                            k: executed_tool[k]
                            for k in ("index", "output")
                            if k in executed_tool
                        }
                    )
                else:
                    additional_kwargs["executed_tools"].append(
                        {k: executed_tool[k] for k in executed_tool if k != "output"}
                    )
        if usage := (chunk.get("x_groq") or {}).get("usage"):
            usage_metadata = _create_usage_metadata(usage)
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
            response_metadata={"model_provider": "groq"},
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    return default_class(content=content)  # type: ignore[call-arg]


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.

    """
    id_ = _dict.get("id")
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict = {}
        if reasoning := _dict.get("reasoning"):
            additional_kwargs["reasoning_content"] = reasoning
        if executed_tools := _dict.get("executed_tools"):
            additional_kwargs["executed_tools"] = executed_tools
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            # Groq sends 'null' (JSON null) for tools with no arguments, but we
            # expect '{}' (empty JSON object) to represent empty arguments
            for raw_tool_call in raw_tool_calls:
                if (
                    raw_tool_call.get("function")
                    and raw_tool_call["function"].get("arguments") == "null"
                ):
                    raw_tool_call["function"]["arguments"] = "{}"
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # pylint: disable=broad-except
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        return AIMessage(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={"model_provider": "groq"},
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def _lc_tool_call_to_groq_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_groq_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _create_usage_metadata(groq_token_usage: dict) -> UsageMetadata:
    """Create usage metadata from Groq token usage response.

    Args:
        groq_token_usage: Token usage dict from Groq API response.

    Returns:
        Usage metadata dict with input/output token details.
    """
    # Support both formats: new Responses API uses "input_tokens",
    # Chat Completions API uses "prompt_tokens"
    input_tokens = (
        groq_token_usage.get("input_tokens")
        or groq_token_usage.get("prompt_tokens")
        or 0
    )
    output_tokens = (
        groq_token_usage.get("output_tokens")
        or groq_token_usage.get("completion_tokens")
        or 0
    )
    total_tokens = groq_token_usage.get("total_tokens") or input_tokens + output_tokens

    # Support both formats for token details:
    # Responses API uses "*_tokens_details", Chat Completions API might use
    # "prompt_token_details"
    input_details_dict = (
        groq_token_usage.get("input_tokens_details")
        or groq_token_usage.get("prompt_tokens_details")
        or {}
    )
    output_details_dict = (
        groq_token_usage.get("output_tokens_details")
        or groq_token_usage.get("completion_tokens_details")
        or {}
    )

    input_token_details: dict = {
        "cache_read": input_details_dict.get("cached_tokens"),
    }
    output_token_details: dict = {
        "reasoning": output_details_dict.get("reasoning_tokens"),
    }
    usage_metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    if filtered_input := {k: v for k, v in input_token_details.items() if v}:
        usage_metadata["input_token_details"] = InputTokenDetails(**filtered_input)  # type: ignore[typeddict-item]
    if filtered_output := {k: v for k, v in output_token_details.items() if v}:
        usage_metadata["output_token_details"] = OutputTokenDetails(**filtered_output)  # type: ignore[typeddict-item]
    return usage_metadata
