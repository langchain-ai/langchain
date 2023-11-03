from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

from requests.exceptions import HTTPError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import (
    BaseChatModel,
    _generate_from_stream,
)
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
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def convert_message_to_dict(message: BaseMessage) -> dict:
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
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
    length: int,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk.

    As the low level API implement is different from openai and other llm.
    Stream response of Tongyi is not split into chunks, but all data generated before.
    For example, the answer 'Hi Pickle Rick! How can I assist you today?'
    Other llm will stream answer:
    'Hi Pickle',
    ' Rick!',
    ' How can I assist you today?'.

    Tongyi answer:
    'Hi Pickle',
    'Hi Pickle Rick!',
    'Hi Pickle Rick! How can I assist you today?'.

    As the GenerationChunk is implemented with chunks. Only return full_text[length:]
    for new chunk.
    """
    full_text = stream_response["output"]["text"]
    text = full_text[length:]
    finish_reason = stream_response["output"].get("finish_reason", None)

    return GenerationChunk(
        text=text,
        generation_info=dict(
            finish_reason=finish_reason,
        ),
    )


def _create_retry_decorator(
    llm: ChatTongyi,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
) -> Callable[[Any], Any]:
    def _before_sleep(retry_state: RetryCallState) -> None:
        if run_manager:
            run_manager.on_retry(retry_state)
        return None

    min_seconds = 1
    max_seconds = 4
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=_before_sleep,
    )


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any],
    default_class: Type[BaseMessageChunk],
    length: int,
) -> BaseMessageChunk:
    role = _dict.get("role")
    full_content = _dict.get("content") or ""
    content = full_content[length:]
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


class ChatTongyi(BaseChatModel):
    """Alibaba Tongyi Qwen chat models API.

    To use, you should have the ``dashscope`` python package installed,
    and set env ``DASHSCOPE_API_KEY`` with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.chat_models import Tongyi
            Tongyi_chat = ChatTongyi()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"dashscope_api_key": "DASHSCOPE_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field(default="qwen-turbo", alias="model")

    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    top_p: float = 0.8
    """Total probability mass of tokens to consider at each step."""

    dashscope_api_key: Optional[str] = None
    """Dashscope api key provide by alicloud."""

    n: int = 1
    """How many completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    max_retries: int = 10
    """Maximum number of retries to make when generating."""

    prefix_messages: List = Field(default_factory=list)
    """Series of messages for Chat input."""

    result_format: str = Field(default="message")
    """Return result format"""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tongyi"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        get_from_dict_or_env(values, "dashscope_api_key", "DASHSCOPE_API_KEY")
        try:
            import dashscope
        except ImportError:
            raise ImportError(
                "Could not import dashscope python package. "
                "Please install it with `pip install dashscope --upgrade`."
            )
        try:
            values["client"] = dashscope.Generation
        except AttributeError:
            raise ValueError(
                "`dashscope` has no `Generation` attribute, this is likely "
                "due to an old version of the dashscope package. Try upgrading it "
                "with `pip install --upgrade dashscope`."
            )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "top_p": self.top_p,
            "stream": self.streaming,
            "n": self.n,
            "result_format": self.result_format,
            **self.model_kwargs,
        }

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**_kwargs: Any) -> Any:
            resp = self.client.call(**_kwargs)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in [400, 401]:
                raise ValueError(
                    f"status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}"
                )
            else:
                raise HTTPError(
                    f"HTTP error occurred: status_code: {resp.status_code} \n "
                    f"code: {resp.code} \n message: {resp.message}",
                    response=resp,
                )

        return _completion_with_retry(**kwargs)

    def stream_completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _stream_completion_with_retry(**_kwargs: Any) -> Any:
            return self.client.call(**_kwargs)

        return _stream_completion_with_retry(**kwargs)

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
            return _generate_from_stream(stream_iter)

        if not messages:
            raise ValueError("No messages provided.")

        message_dicts, params = self._create_message_dicts(messages, stop)

        if message_dicts[-1]["role"] != "user":
            raise ValueError("Last message should be user message.")

        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        # Mark current chunk total length
        length = 0
        default_chunk_class = AIMessageChunk
        for chunk in self.stream_completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk["output"]["choices"]) == 0:
                continue
            choice = chunk["output"]["choices"][0]

            chunk = _convert_delta_to_message_chunk(
                choice["message"], default_chunk_class, length
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            length = len(choice["message"]["content"])

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params()

        # Ensure `stop` is a list of strings
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        creds: Dict[str, Any] = {
            "api_key": self.dashscope_api_key,
        }
        return {**self._default_params, **creds}

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["output"]["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)
