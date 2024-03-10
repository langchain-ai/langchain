from __future__ import annotations

import importlib.util
import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core._api import beta
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.constants import ENDPOINT as DEFAULT_MISTRAL_ENDPOINT
from mistralai.exceptions import (
    MistralAPIException,
    MistralConnectionException,
    MistralException,
)
from mistralai.models.chat_completion import (
    ChatCompletionResponse as MistralChatCompletionResponse,
)
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage
from mistralai.models.chat_completion import DeltaMessage as MistralDeltaMessage

logger = logging.getLogger(__name__)


def _create_retry_decorator(
    llm: ChatMistralAI,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions"""

    errors = [
        MistralException,
        MistralAPIException,
        MistralConnectionException,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


def _convert_mistral_chat_message_to_message(
    _message: MistralChatMessage,
) -> BaseMessage:
    role = _message.role
    content = cast(Union[str, List], _message.content)
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs: Dict = {}
        if hasattr(_message, "tool_calls") and getattr(_message, "tool_calls"):
            additional_kwargs["tool_calls"] = [
                tc.model_dump() for tc in getattr(_message, "tool_calls")
            ]
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(content=content, name=_message.name)  # type: ignore[attr-defined]
    else:
        return ChatMessage(content=content, role=role)


async def acompletion_with_retry(
    llm: ChatMistralAI,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        stream = kwargs.pop("stream", False)
        if stream:
            return llm.async_client.chat_stream(**kwargs)
        else:
            return await llm.async_client.chat(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_delta_to_message_chunk(
    _delta: MistralDeltaMessage, default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = getattr(_delta, "role")
    content = getattr(_delta, "content", "")
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        additional_kwargs: Dict = {}
        if hasattr(_delta, "tool_calls") and getattr(_delta, "tool_calls"):
            additional_kwargs["tool_calls"] = [
                tc.model_dump() for tc in getattr(_delta, "tool_calls")
            ]
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_message_to_mistral_chat_message(
    message: BaseMessage,
) -> MistralChatMessage:
    if isinstance(message, ChatMessage):
        mistral_message = MistralChatMessage(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        mistral_message = MistralChatMessage(role="user", content=message.content)
    elif isinstance(message, AIMessage):
        if "tool_calls" in message.additional_kwargs:
            from mistralai.models.chat_completion import (  # type: ignore[attr-defined]
                ToolCall as MistralToolCall,
            )

            tool_calls = [
                MistralToolCall.model_validate(tc)
                for tc in message.additional_kwargs["tool_calls"]
            ]
        else:
            tool_calls = None
        mistral_message = MistralChatMessage(
            role="assistant", content=message.content, tool_calls=tool_calls
        )
    elif isinstance(message, SystemMessage):
        mistral_message = MistralChatMessage(role="system", content=message.content)
    elif isinstance(message, ToolMessage):
        mistral_message = MistralChatMessage(
            role="tool", content=message.content, name=message.name
        )
    else:
        raise ValueError(f"Got unknown type {message}")
    return mistral_message


class ChatMistralAI(BaseChatModel):
    """A chat model that uses the MistralAI API."""

    client: MistralClient = Field(default=None)  #: :meta private:
    async_client: MistralAsyncClient = Field(default=None)  #: :meta private:
    mistral_api_key: Optional[SecretStr] = None
    endpoint: str = DEFAULT_MISTRAL_ENDPOINT
    max_retries: int = 5
    timeout: int = 120
    max_concurrent_requests: int = 64

    model: str = "mistral-small"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    random_seed: Optional[int] = None
    safe_mode: bool = False

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the API."""
        defaults = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "random_seed": self.random_seed,
            "safe_mode": self.safe_mode,
        }
        filtered = {k: v for k, v in defaults.items() if v is not None}
        return filtered

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the client."""
        return self._default_params

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            stream = kwargs.pop("stream", False)
            if stream:
                return self.client.chat_stream(**kwargs)
            else:
                return self.client.chat(**kwargs)

        return _completion_with_retry(**kwargs)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists, temperature, and top_p."""
        mistralai_spec = importlib.util.find_spec("mistralai")
        if mistralai_spec is None:
            raise MistralException(
                "Could not find mistralai python package. "
                "Please install it with `pip install mistralai`"
            )

        values["mistral_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "mistral_api_key", "MISTRAL_API_KEY", default=""
            )
        )
        values["client"] = MistralClient(
            api_key=values["mistral_api_key"].get_secret_value(),
            endpoint=values["endpoint"],
            max_retries=values["max_retries"],
            timeout=values["timeout"],
        )
        values["async_client"] = MistralAsyncClient(
            api_key=values["mistral_api_key"].get_secret_value(),
            endpoint=values["endpoint"],
            max_retries=values["max_retries"],
            timeout=values["timeout"],
            max_concurrent_requests=values["max_concurrent_requests"],
        )

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_chat_result(
        self, response: MistralChatCompletionResponse
    ) -> ChatResult:
        generations = []
        for res in response.choices:
            finish_reason = getattr(res, "finish_reason")
            if finish_reason:
                finish_reason = finish_reason.value
            gen = ChatGeneration(
                message=_convert_mistral_chat_message_to_message(res.message),
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)
        token_usage = getattr(response, "usage")
        token_usage = vars(token_usage) if token_usage else {}
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[MistralChatMessage], Dict[str, Any]]:
        params = self._client_params
        if stop is not None or "stop" in params:
            if "stop" in params:
                params.pop("stop")
            logger.warning(
                "Parameter `stop` not yet supported (https://docs.mistral.ai/api)"
            )
        message_dicts = [_convert_message_to_mistral_chat_message(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            if run_manager:
                run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)
            yield ChatGenerationChunk(message=chunk)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @beta()
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_mistralai import ChatMistralAI
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_mistralai import ChatMistralAI
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_mistralai import ChatMistralAI
                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        llm = self.bind_tools([schema], tool_choice="any")
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
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

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "mistralai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"mistral_api_key": "MISTRAL_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "mistralai"]
