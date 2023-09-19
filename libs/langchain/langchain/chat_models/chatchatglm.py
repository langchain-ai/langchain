"""ChatGLM chat wrapper."""
from __future__ import annotations

import copy
import logging
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import create_base_retry_decorator
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
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
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError:
        raise ValueError(
            "Could not import tiktoken python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install tiktoken`."
        )
    return tiktoken


def _create_retry_decorator(
    llm: ChatChatGLM,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    errors = [
        BaseException,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


async def acompletion_with_retry(
    llm: ChatChatGLM,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use ChatGLM's async api
        # https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_pro/invoke
        m_kwargs = copy.deepcopy(kwargs)
        m_kwargs["prompt"] = kwargs["messages"]
        if len(m_kwargs["prompt"]) // 2 == 0:
            raise ValueError("The length of the Prompt must be an odd number.")
        if m_kwargs.get("streaming") or m_kwargs.get("stream"):
            try:
                from aiostream.stream import list as alist
            except ImportError as e:
                raise ImportError(
                    "Streaming with ChatChatGLMrequires optional dependency aiostream. "
                    "To install please run `pip install aiostream`."
                ) from e

            async def async_gen(**m_kwargs: Any) -> Any:
                for event in llm.client.sse_invoke(**m_kwargs).events():
                    yield event.data

            return alist(async_gen(**m_kwargs))
        else:
            return llm.client.invoke(**m_kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        # Fix for azure
        # Also ChatGLM returns None for tool invocations
        content = _dict.get("content", "") or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=eval(content), additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def convert_chatglm_messages(messages: List[dict]) -> List[BaseMessage]:
    """Convert dictionaries representing ChatGLM messages to LangChain format.

    Args:
        messages: List of dictionaries representing ChatGLM messages

    Returns:
        List of LangChain BaseMessage objects.
    """
    return [_convert_dict_to_message(m) for m in messages]


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatChatGLM(BaseChatModel):
    """Wrapper around ChatGLM Chat large language models.

    To use, you should have the ``zhipuai`` python package installed, and the
    environment variable ``CHATGLM_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the chatglm.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatChatGLM
            chatglm = ChatChatGLM(model_name="gpt-3.5-turbo")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"chatglm_api_key": "CHATGLM_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any = None  #: :meta private:
    model_name: str = Field(default="chatglm_pro", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    chatglm_api_key: Optional[str] = None
    """Base URL path for API requests, 
    leave blank if not using a proxy or service emulator."""
    chatglm_api_base: Optional[str] = None
    chatglm_organization: Optional[str] = None
    # to support explicit proxy for ChatGLM
    chatglm_proxy: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to ChatGLM completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    tiktoken_model_name: Optional[str] = None
    """The model name to pass to tiktoken when using this class. 
    Tiktoken is used to count the number of tokens in documents to constrain 
    them to be under a certain limit. By default, when set to None, this will 
    be the same as the embedding model name. However, there are some cases 
    where you may want to use this Embedding class with a model name not 
    supported by tiktoken. This can include when using Azure embeddings or 
    when using one of the many model providers that expose an ChatGLM-like 
    API but with different models. In those cases, in order to avoid erroring 
    when tiktoken is called, you can specify a model name to use here."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["chatglm_api_key"] = get_from_dict_or_env(
            values, "chatglm_api_key", "CHATGLM_API_KEY"
        )
        values["chatglm_organization"] = get_from_dict_or_env(
            values,
            "chatglm_organization",
            "CHATGLM_ORGANIZATION",
            default="",
        )
        values["chatglm_api_base"] = get_from_dict_or_env(
            values,
            "chatglm_api_base",
            "CHATGLM_API_BASE",
            default="",
        )
        values["chatglm_proxy"] = get_from_dict_or_env(
            values,
            "chatglm_proxy",
            "CHATGLM_PROXY",
            default="",
        )
        try:
            import zhipuai

            zhipuai.api_key = values["chatglm_api_key"]
        except ImportError:
            raise ValueError(
                "Could not import zhipuai python package. "
                "Please install it with `pip install zhipuai`."
            )
        try:
            values["client"] = zhipuai.model_api
        except AttributeError:
            raise ValueError(
                "`zhipuai` has no `model_api` attribute, this is likely "
                "due to an old version of the zhipuai package. Try upgrading it "
                "with `pip install --upgrade zhipuai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ChatGLM API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            m_kwargs = copy.deepcopy(kwargs)
            m_kwargs["prompt"] = kwargs["messages"]
            if len(m_kwargs["prompt"]) // 2 == 0:
                raise ValueError("The length of the Prompt must be an odd number.")
            if m_kwargs.get("streaming") or m_kwargs.get("stream"):
                return self.client.sse_invoke(**m_kwargs)
            else:
                return self.client.invoke(**m_kwargs)

        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

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
        for event in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ).events():
            delta = {"role": "assistant", "content": event.data}
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stream if stream is not None else self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["data"]["choices"]:
            message = _convert_dict_to_message(res)
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

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
        async for event in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(event):
                delta = {"role": "assistant", "content": event[-1]}
            else:
                delta = {"role": "assistant", "content": ""}
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            yield ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stream if stream is not None else self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the chatglm client."""
        chatglm_creds: Dict[str, Any] = {
            "api_key": self.chatglm_api_key,
            "api_base": self.chatglm_api_base,
            "organization": self.chatglm_organization,
            "model": self.model_name,
        }
        if self.chatglm_proxy:
            import zhipuai

            zhipuai.api_key = self.chatglm_api_key
            # zhipuai.proxy = {"http": self.chatglm_proxy, "https": self.chatglm_proxy}
            # type: ignore[assignment]  # noqa: E501
        return {**self._default_params, **chatglm_creds}

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
        return "chatglm-chat"

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        tiktoken_ = _import_tiktoken()
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://open.bigmodel.cn/dev/api
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("chatglm_pro"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("chatglm_std") or model.startswith("chatglm_lite"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://open.bigmodel.cn/dev/api for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens
