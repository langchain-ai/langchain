"""OpenAI chat wrapper."""

from __future__ import annotations

import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import zhipuai
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.load import dumpd, dumps
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
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.utils import build_extra_kwargs

from langchain_glm.chat_models.all_tools_message import (
    ALLToolsMessageChunk,
    _paser_chunk,
)

if TYPE_CHECKING:
    from langchain_core.pydantic_v1 import BaseModel
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
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
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
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
    elif default_class == ALLToolsMessageChunk:
        return ALLToolsMessageChunk(
            content=content, additional_kwargs=additional_kwargs
        )
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore


class _FunctionCall(TypedDict):
    name: str


class ChatZhipuAI(BaseChatModel):
    """ZhipuAI chat model integration.

    Setup:
        Install ``langchain-zhipuai`` and set environment variable ``ZHIPUAI_API_KEY``

        .. code-block:: bash

            pip install langchain-zhipuai -U
            export ZHIPUAI_API_KEY="your-api-key"

    Key init args — completion params:
        model: Optional[str]
            Name of OpenAI model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        api_key: Optional[str]
        ZhipuAI API key. If not passed in will be read from env var ZHIPUAI_API_KEY.
        api_base: Optional[str]
        Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatZhipuAI

            zhipuai_chat = ChatZhipuAI(
                temperature=0.5,
                api_key="your-api-key",
                model="glm-4",
                # api_base="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
                ("human", "我喜欢编程。"),
            ]
            zhipuai_chat.invoke(messages)

        .. code-block:: python

            AIMessage(content='I enjoy programming.', response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 23, 'total_tokens': 29}, 'model_name': 'glm-4', 'finish_reason': 'stop'}, id='run-c5d9af91-55c6-470e-9545-02b2fa0d7f9d-0')

    Stream:
        .. code-block:: python

            for chunk in zhipuai_chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='I' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content=' enjoy' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content=' programming' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content='.' id='run-4df71729-618f-4e2b-a4ff-884682723082'
            content='' response_metadata={'finish_reason': 'stop'} id='run-4df71729-618f-4e2b-a4ff-884682723082'

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block::

            AIMessageChunk(content='I enjoy programming.', response_metadata={'finish_reason': 'stop'}, id='run-20b05040-a0b4-4715-8fdc-b39dba9bfb53')

    Async:
        .. code-block:: python

            await zhipuai_chat.ainvoke(messages)

            # stream:
            # async for chunk in zhipuai_chat.astream(messages):
            #    print(chunk)


        .. code-block:: python

            [AIMessage(content='I enjoy programming.', response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 23, 'total_tokens': 29}, 'model_name': 'glm-4', 'finish_reason': 'stop'}, id='run-ba06af9d-4baa-40b2-9298-be9c62aa0849-0')]

    Response metadata
        .. code-block:: python

            ai_msg = zhipuai_chat.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'token_usage': {'completion_tokens': 6,
              'prompt_tokens': 23,
              'total_tokens': 29},
              'model_name': 'glm-4',
              'finish_reason': 'stop'}

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "openai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.zhipuai_api_base:
            attributes["zhipuai_api_base"] = self.zhipuai_api_base

        if self.zhipuai_proxy:
            attributes["zhipuai_proxy"] = self.zhipuai_proxy

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="glm-4", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    zhipuai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `ZHIPUAI_API_KEY` if not provided."""
    zhipuai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    # to support explicit proxy for OpenAI
    zhipuai_proxy: Optional[str] = None
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True, allow_reuse=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        values["zhipuai_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "zhipuai_api_key", "ZHIPUAI_API_KEY")
        )

        values["zhipuai_api_base"] = values["zhipuai_api_base"] or os.getenv(
            "ZHIPUAI_API_BASE"
        )
        values["zhipuai_proxy"] = get_from_dict_or_env(
            values,
            "zhipuai_proxy",
            "ZHIPUAI_PROXY",
            default="",
        )

        client_params = {
            "api_key": (
                values["zhipuai_api_key"].get_secret_value()
                if values["zhipuai_api_key"]
                else None
            ),
            "base_url": values["zhipuai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "http_client": values["http_client"],
        }

        if not values.get("client"):
            values["client"] = zhipuai.ZhipuAI(**client_params).chat.completions

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        return combined

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        if type(self)._stream == BaseChatModel._stream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, config=config, stop=stop, **kwargs)
            )
        else:
            config = ensure_config(config)
            messages = self._convert_input(input).to_messages()
            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs}
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                dumpd(self),
                [messages],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )
            generation: Optional[ChatGenerationChunk] = None
            try:
                for chunk in self._stream(messages, stop=stop, **kwargs):
                    if chunk.message.id is None:
                        chunk.message.id = f"run-{run_manager.run_id}"
                    chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                    if (
                        isinstance(chunk.message, ALLToolsMessageChunk)
                        and chunk.message.content == ""
                    ):
                        tool_calls, invalid_tool_calls = _paser_chunk(
                            chunk.message.tool_call_chunks
                        )

                        for chunk_tool in invalid_tool_calls:
                            if isinstance(chunk_tool["args"], str):
                                args_ = parse_partial_json(chunk_tool["args"])
                            else:
                                args_ = chunk_tool["args"]
                            if not isinstance(args_, dict):
                                raise ValueError("Malformed args.")
                            if "input" in args_:
                                run_manager.on_llm_new_token(
                                    cast(str, args_["input"]), chunk=chunk
                                )

                    else:
                        run_manager.on_llm_new_token(
                            cast(str, chunk.message.content), chunk=chunk
                        )
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        if (
            type(self)._astream is BaseChatModel._astream
            and type(self)._stream is BaseChatModel._stream
        ):
            # No async or sync stream is implemented, so fall back to ainvoke
            yield cast(
                BaseMessageChunk,
                await self.ainvoke(input, config=config, stop=stop, **kwargs),
            )
            return

        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **kwargs}
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            config.get("metadata"),
            self.metadata,
        )
        (run_manager,) = await callback_manager.on_chat_model_start(
            dumpd(self),
            [messages],
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        generation: Optional[ChatGenerationChunk] = None
        try:
            async for chunk in self._astream(
                messages,
                stop=stop,
                **kwargs,
            ):
                if chunk.message.id is None:
                    chunk.message.id = f"run-{run_manager.run_id}"
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if (
                    isinstance(chunk.message, ALLToolsMessageChunk)
                    and chunk.message.content == ""
                ):
                    tool_calls, invalid_tool_calls = _paser_chunk(
                        chunk.message.tool_call_chunks
                    )

                    for chunk_tool in invalid_tool_calls:
                        if isinstance(chunk_tool["args"], str):
                            try:
                                args_ = parse_partial_json(chunk_tool["args"])
                            except Exception as e:
                                args_ = {"input": chunk_tool["args"]}
                        else:
                            args_ = chunk_tool["args"]
                        if not isinstance(args_, dict):
                            raise ValueError("Malformed args.")
                        if "input" in args_:
                            await run_manager.on_llm_new_token(
                                cast(str, args_["input"]), chunk=chunk
                            )
                        else:
                            await run_manager.on_llm_new_token(
                                cast(str, args_), chunk=chunk
                            )
                else:
                    await run_manager.on_llm_new_token(
                        cast(str, chunk.message.content), chunk=chunk
                    )
                yield chunk.message
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
        except BaseException as e:
            await run_manager.on_llm_error(
                e,
                response=LLMResult(generations=[[generation]] if generation else []),
            )
            raise e
        else:
            await run_manager.on_llm_end(
                LLMResult(generations=[[generation]]),
            )

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
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            # all_tools chunk load action exec parse tool
            if params["model"] in [
                "glm-4-alltools-dev",
                "tob-alltools-api-dev",
                "glm-4-alltools",
            ]:
                default_chunk_class = ALLToolsMessageChunk

            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model_name,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "zhipuai-chat"

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        function_call: Optional[
            Union[_FunctionCall, str, Literal["auto", "none"]]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind functions (and other objects) to this chat model.

        Assumes model is compatible with OpenAI function-calling API.

        NOTE: Using bind_tools is recommended instead, as the `functions` and
            `function_call` request parameters are officially marked as deprecated by
            OpenAI.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            function_call = (
                {"name": function_call}
                if isinstance(function_call, str)
                and function_call not in ("auto", "none")
                else function_call
            )
            if isinstance(function_call, dict) and len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if (
                isinstance(function_call, dict)
                and formatted_functions[0]["name"] != function_call["name"]
            ):
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            kwargs = {**kwargs, "function_call": function_call}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"]]] = None,
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
        if tool_choice is not None:
            if isinstance(tool_choice, str) and (tool_choice not in ("auto", "none")):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and (len(formatted_tools) != 1):
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, dict) and (
                formatted_tools[0]["function"]["name"]
                != tool_choice["function"]["name"]
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tool was {formatted_tools[0]['function']['name']}."
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)


def _gen_info_and_msg_metadata(
    generation: Union[ChatGeneration, ChatGenerationChunk],
) -> dict:
    return {
        **(generation.generation_info or {}),
        **generation.message.response_metadata,
    }
