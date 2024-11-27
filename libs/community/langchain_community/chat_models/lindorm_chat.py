"""Lindorm AI chat model."""

from __future__ import annotations

import logging
from collections import deque
from importlib import util
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
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
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from pydantic import model_validator
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1
DEFAULT_NUM_OUTPUTS = 256  # tokens

HTTP_HDR_AK_KEY = "x-ld-ak"
HTTP_HDR_SK_KEY = "x-ld-sk"
REST_URL_PATH = "/v1/ai"
REST_URL_MODELS_PATH = REST_URL_PATH + "/models"
INFER_INPUT_KEY = "input"
INFER_PARAMS_KEY = "params"
RSP_DATA_KEY = "data"
RSP_MODELS_KEY = "models"


def convert_message_chunk_to_message(message_chunk: BaseMessageChunk) -> BaseMessage:
    if isinstance(message_chunk, HumanMessageChunk):
        return HumanMessage(content=message_chunk.content)
    elif isinstance(message_chunk, AIMessageChunk):
        return AIMessage(content=message_chunk.content)
    elif isinstance(message_chunk, SystemMessageChunk):
        return SystemMessage(content=message_chunk.content)
    elif isinstance(message_chunk, ChatMessageChunk):
        return ChatMessage(role=message_chunk.role, content=message_chunk.content)
    else:
        raise TypeError(f"Got unknown type {message_chunk}")


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dict."""
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _create_retry_decorator(llm: ChatLindormAI) -> Callable[[Any], Any]:
    min_seconds = 1
    max_seconds = 4
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(HTTPError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


class ChatLindormAI(BaseChatModel):
    """LindormAI chat models API.

    To use, you should have the ``lindormai`` python package installed,
    and set variable ``endpoint``, ``username``, ``password`` and ``model_name``.

    Example:
        .. code-block:: python
            from langchain_community.chat_models.lindormai import ChatLindormAI
            lindorm_ai_chat = ChatLindormAI(
                endpoint='https://ld-xxx-proxy-ml.lindorm.rds.aliyuncs.com:9002',
                username='root',
                password='xxx',
                model_name='qwen-72b'
            )
    """

    client: Any  #: :meta private:
    endpoint: str = Field(...)
    username: str = Field(...)
    password: str = Field(...)
    model_name: str = Field(...)

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.",
        default=DEFAULT_NUM_OUTPUTS,
        gt=0,
    )
    temperature: Optional[float] = Field(
        description="The temperature to use during generation.",
        default=DEFAULT_TEMPERATURE,
        gte=0.0,
        lte=2.0,
    )
    top_k: Optional[int] = Field(
        description="Sample counter when generate.", default=None
    )
    top_p: Optional[float] = Field(
        description="Sample probability threshold when generate.", default=None
    )
    seed: Optional[int] = Field(
        description="Random seed when generate.", default=1234, gte=0
    )
    repetition_penalty: Optional[float] = Field(
        description="Penalty for repeated words in generated text; 1.0 is no penalty.",
        default=None,
    )
    max_retries: int = 10
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "lindormai"

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Ensure the client is initialized properly."""
        if not values.get("client"):
            if values.get("username") is None:
                raise ValueError("username can't be empty")
            if values.get("password") is None:
                raise ValueError("password can't be empty")
            if values.get("endpoint") is None:
                raise ValueError("endpoint can't be empty")
        return values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
    def __post_with_retry(
        self, url: str, data: Any = None, json: Any = None, **kwargs: Any
    ) -> Any:
        response = requests.post(url=url, data=data, json=json, **kwargs)
        response.raise_for_status()
        return response

    def _infer(self, model_name: str, input_data: Any, params: Any) -> Any:
        url = f"{self.endpoint}{REST_URL_MODELS_PATH}/{model_name}/infer"
        infer_dict = {INFER_INPUT_KEY: input_data, INFER_PARAMS_KEY: params}
        result = None
        try:
            headers = {HTTP_HDR_AK_KEY: self.username, HTTP_HDR_SK_KEY: self.password}
            response = self.__post_with_retry(url, json=infer_dict, headers=headers)
            response.raise_for_status()
            result = response.json()
        except Exception as error:
            logger.error(
                f"infer model for {model_name} with "
                f"input {input_data} and params {params}: {error}"
            )
        return result[RSP_DATA_KEY] if result else None

    def __stream_infer(self, model_name: str, input_data: Any, params: Any) -> Any:
        # 动态查找 ijson 模块
        ijson_spec = util.find_spec("ijson")

        if ijson_spec is not None:
            import ijson  # 仅在模块存在的情况下进行导入
        else:
            raise ImportError("Module 'ijson' is not available")

        @ijson.coroutine
        def recv_output(output_q: Any) -> Any:
            while True:
                item = yield
                output_q.append(item)

        url = f"{self.endpoint}{REST_URL_MODELS_PATH}/{model_name}/infer"
        infer_dict = {INFER_INPUT_KEY: input_data, INFER_PARAMS_KEY: params}
        output_q: Deque = deque()
        try:
            headers = {HTTP_HDR_AK_KEY: self.username, HTTP_HDR_SK_KEY: self.password}
            with self.__post_with_retry(
                url, json=infer_dict, headers=headers, stream=True
            ) as response:
                response.raise_for_status()

                ijson_coro = ijson.items_coro(
                    recv_output(output_q), "data.outputs.item"
                )
                for chunk in response.iter_content(
                    chunk_size=1024, decode_unicode=True
                ):
                    logger.debug(
                        f"stream infer model for {model_name} with "
                        f"input {input_data} and params {params} result: {chunk}"
                    )
                    ijson_coro.send(chunk.encode())
                    while output_q:
                        yield output_q.popleft()
                ijson_coro.close()
        except Exception as error:
            logger.error(
                f"stream infer model for {model_name} with "
                f"input {input_data} and params {params}: {error}"
            )

    def _stream_infer(self, model_name: str, input_data: Any, params: Any) -> Any:
        if params is None:
            params = {}
        yield from self.__stream_infer(model_name, input_data, params)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Lindorm AI SDK."""
        default_params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.top_k is not None:
            default_params["top_k"] = self.top_k
        if self.top_p is not None:
            default_params["top_p"] = self.top_p
        return default_params

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _completion_with_retry(**_kwargs: Any) -> Any:
            return self._infer(
                model_name=self.model_name,
                input_data=str(_kwargs["input_data"]),
                params=_kwargs,
            )

        return _completion_with_retry(**kwargs)

    def stream_completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _stream_completion_with_retry(**_kwargs: Any) -> Any:
            responses = self._stream_infer(
                model_name=self.model_name,
                input_data=str(_kwargs["input_data"]),
                params=_kwargs,
            )
            for resp in responses:
                yield resp

        return _stream_completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        generations = []
        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            generations.append(self._chunk_to_generation(generation))
        else:
            params: Dict[str, Any] = self._invocation_params(
                messages=messages, stop=stop, **kwargs
            )
            params["force_nonstream"] = True
            resp = self.completion_with_retry(**params)
            generations.append(
                ChatGeneration(**self._chat_generation_from_lindormai_resp(resp))
            )
        return ChatResult(
            generations=generations,
            llm_output={
                "model_name": self.model_name,
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "Please use `_generate`. Official does not support asynchronous requests"
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            messages=messages, stop=stop, stream=True, **kwargs
        )
        for stream_resp in self.stream_completion_with_retry(**params):
            chunk = ChatGenerationChunk(
                **self._chat_generation_from_lindormai_resp(stream_resp, is_chunk=True)
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:
    #     raise NotImplementedError(
    #         "Please use `_stream`. Official does not support asynchronous requests"
    #     )

    def _invocation_params(
        self, messages: List[BaseMessage], stop: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        if params.get("stream"):
            params["stream"] = True
        message_dicts = [convert_message_to_dict(m) for m in messages]
        # According to the docs, the last message should be a `user` message
        if message_dicts[-1]["role"] != "user":
            raise ValueError("Last message should be a user message.")
        # And the `system` message should be the first message if present
        system_message_indices = [
            i for i, m in enumerate(message_dicts) if m["role"] == "system"
        ]
        if len(system_message_indices) == 1 and system_message_indices[0] != 0:
            raise ValueError("System message can only be the first message.")
        elif len(system_message_indices) > 1:
            raise ValueError("There can only be one system message at most.")
        params["input_data"] = message_dicts
        return params

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        if llm_outputs[0] is None:
            return {}
        return llm_outputs[0]

    @staticmethod
    def _chat_generation_from_lindormai_resp(
        resp: Any, is_chunk: bool = False
    ) -> Dict[str, Any]:
        if resp is None:
            raise ValueError("Response cannot be None")
        elif "output" in resp:
            content = resp["output"]
        else:
            content = ""
            for output in resp["outputs"]:
                content += output
        return dict(message=AIMessage(content=content))

    @staticmethod
    def _chunk_to_generation(chunk: ChatGenerationChunk) -> ChatGeneration:
        return ChatGeneration(
            message=convert_message_chunk_to_message(chunk.message),
            generation_info=chunk.generation_info,
        )

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        from langchain.chains.openai_functions.base import convert_to_openai_function

        formatted_functions = [convert_to_openai_function(fn) for fn in functions]

        if function_call is not None:
            if len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if formatted_functions[0]["name"] != function_call:
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            function_call_ = {"name": function_call}
            kwargs = {**kwargs, "function_call": function_call_}

        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )
