"""Wrapper around Perplexity APIs."""

from __future__ import annotations

import logging
from operator import itemgetter
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils import from_env, get_pydantic_field_names
from langchain_core.utils.pydantic import (
    is_basemodel_subclass,
)
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from typing_extensions import Self

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]

logger = logging.getLogger(__name__)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _create_usage_metadata(token_usage: dict) -> UsageMetadata:
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


@deprecated(
    since="0.3.21",
    removal="1.0",
    alternative_import="langchain_perplexity.ChatPerplexity",
)
class ChatPerplexity(BaseChatModel):
    """`Perplexity AI` Chat models API.

    Setup:
        To use, you should have the ``openai`` python package installed, and the
        environment variable ``PPLX_API_KEY`` set to your API key.
        Any parameters that are valid to be passed to the openai.create call
        can be passed in, even if not explicitly saved on this class.

        .. code-block:: bash

            pip install openai
            export PPLX_API_KEY=your_api_key

        Key init args - completion params:
            model: str
                Name of the model to use. e.g. "llama-3.1-sonar-small-128k-online"
            temperature: float
                Sampling temperature to use. Default is 0.7
            max_tokens: Optional[int]
                Maximum number of tokens to generate.
            streaming: bool
                Whether to stream the results or not.

        Key init args - client params:
            pplx_api_key: Optional[str]
                API key for PerplexityChat API. Default is None.
            request_timeout: Optional[Union[float, Tuple[float, float]]]
                Timeout for requests to PerplexityChat completion API. Default is None.
            max_retries: int
                Maximum number of retries to make when generating.

        See full list of supported init args and their descriptions in the params section.

        Instantiate:
            .. code-block:: python

                from langchain_community.chat_models import ChatPerplexity

                llm = ChatPerplexity(
                    model="llama-3.1-sonar-small-128k-online",
                    temperature=0.7,
                )

        Invoke:
            .. code-block:: python

                messages = [
                    ("system", "You are a chatbot."),
                    ("user", "Hello!")
                ]
                llm.invoke(messages)

        Invoke with structured output:
            .. code-block:: python

                from pydantic import BaseModel

                class StructuredOutput(BaseModel):
                    role: str
                    content: str

                llm.with_structured_output(StructuredOutput)
                llm.invoke(messages)

        Invoke with perplexity-specific params:
            .. code-block:: python

                llm.invoke(messages, extra_body={"search_recency_filter": "week"})

        Stream:
            .. code-block:: python

                for chunk in llm.stream(messages):
                    print(chunk.content)

        Token usage:
            .. code-block:: python

                response = llm.invoke(messages)
                response.usage_metadata

        Response metadata:
            .. code-block:: python

                response = llm.invoke(messages)
                response.response_metadata

    """  # noqa: E501

    client: Any = None  #: :meta private:
    model: str = "llama-3.1-sonar-small-128k-online"
    """Model name."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    pplx_api_key: Optional[str] = Field(
        default_factory=from_env("PPLX_API_KEY", default=None), alias="api_key"
    )
    """Base URL path for API requests,
    leave blank if not using a proxy or service emulator."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = Field(
        None, alias="timeout"
    )
    """Timeout for requests to PerplexityChat completion API. Default is None."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not a default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            self.client = openai.OpenAI(
                api_key=self.pplx_api_key, base_url="https://api.perplexity.ai"
            )
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling PerplexityChat API."""
        return {
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _convert_delta_to_message_chunk(
        self, _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _dict.get("role")
        content = _dict.get("content") or ""
        additional_kwargs: Dict = {}
        if _dict.get("function_call"):
            function_call = dict(_dict["function_call"])
            if "name" in function_call and function_call["name"] is None:
                function_call["name"] = ""
            additional_kwargs["function_call"] = function_call
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role == "function" or default_class == FunctionMessageChunk:
            return FunctionMessageChunk(content=content, name=_dict["name"])
        elif role == "tool" or default_class == ToolMessageChunk:
            return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
        else:
            return default_class(content=content)  # type: ignore[call-arg]

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk
        params.pop("stream", None)
        if stop:
            params["stop_sequences"] = stop
        stream_resp = self.client.chat.completions.create(
            messages=message_dicts, stream=True, **params
        )
        first_chunk = True
        prev_total_usage: Optional[UsageMetadata] = None
        for chunk in stream_resp:
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            # Collect standard usage metadata (transform from aggregate to delta)
            if total_usage := chunk.get("usage"):
                lc_total_usage = _create_usage_metadata(total_usage)
                if prev_total_usage:
                    usage_metadata: Optional[UsageMetadata] = {
                        "input_tokens": lc_total_usage["input_tokens"]
                        - prev_total_usage["input_tokens"],
                        "output_tokens": lc_total_usage["output_tokens"]
                        - prev_total_usage["output_tokens"],
                        "total_tokens": lc_total_usage["total_tokens"]
                        - prev_total_usage["total_tokens"],
                    }
                else:
                    usage_metadata = lc_total_usage
                prev_total_usage = lc_total_usage
            else:
                usage_metadata = None
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]

            additional_kwargs = {}
            if first_chunk:
                additional_kwargs["citations"] = chunk.get("citations", [])
                for attr in ["images", "related_questions"]:
                    if attr in chunk:
                        additional_kwargs[attr] = chunk[attr]

            chunk = self._convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )

            if isinstance(chunk, AIMessageChunk) and usage_metadata:
                chunk.usage_metadata = usage_metadata

            if first_chunk:
                chunk.additional_kwargs |= additional_kwargs
                first_chunk = False

            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.completions.create(messages=message_dicts, **params)
        if usage := getattr(response, "usage", None):
            usage_metadata = _create_usage_metadata(usage.model_dump())
        else:
            usage_metadata = None

        additional_kwargs = {"citations": response.citations}
        for attr in ["images", "related_questions"]:
            if hasattr(response, attr):
                additional_kwargs[attr] = getattr(response, attr)

        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs=additional_kwargs,
            usage_metadata=usage_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        pplx_creds: Dict[str, Any] = {
            "model": self.model,
        }
        return {**pplx_creds, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "perplexitychat"

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema for Preplexity.
        Currently, Preplexity only supports "json_schema" method for structured output
        as per their official documentation: https://docs.perplexity.ai/guides/structured-outputs

        Args:
            schema:
                The output schema. Can be passed in as:

                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class

            method: The method for steering model generation, currently only support:

                - "json_schema": Use the JSON Schema to parse the model output


            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            | If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            | If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - "raw": BaseMessage
            - "parsed": None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - "parsing_error": Optional[BaseException]

        """  # noqa: E501
        if method in ("function_calling", "json_mode"):
            method = "json_schema"
        if method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_schema'. "
                    "Received None."
                )
            is_pydantic_schema = _is_pydantic_class(schema)
            if is_pydantic_schema and hasattr(
                schema, "model_json_schema"
            ):  # accounting for pydantic v1 and v2
                response_format = schema.model_json_schema()  # type: ignore[union-attr]
            elif is_pydantic_schema:
                response_format = schema.schema()  # type: ignore[union-attr]
            elif isinstance(schema, dict):
                response_format = schema
            elif type(schema).__name__ == "_TypedDictMeta":
                adapter = TypeAdapter(schema)  # if use passes typeddict
                response_format = adapter.json_schema()

            llm = self.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": response_format},
                }
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected 'json_schema' Received:\
                    '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
