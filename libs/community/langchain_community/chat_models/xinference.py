"""Xinference chat wrapper."""
from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import types
import typing
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain.pydantic_v1 import root_validator
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
from langchain.utils import get_pydantic_field_names

logger = logging.getLogger(__name__)


async def to_thread(func: Callable, /, *args: Any, **kwargs: Any) -> Any:
    """
    Copy from CPython for compatible with Python < 3.9

    Asynchronously run function *func* in a separate thread.

    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.

    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)


async def _to_async_iter(sync_iter: typing.Iterable) -> AsyncIterator:
    """Convert a sync iterator to an async iterator."""
    it = iter(sync_iter)
    done = object()

    def safe_next() -> Any:
        # Converts StopIteration to a sentinel value to avoid:
        # TypeError: StopIteration interacts badly with generators
        # and cannot be raised into a Future
        try:
            return next(it)
        except StopIteration:
            return done

    while True:
        value = await to_thread(safe_next)
        if value is done:
            break
        yield value


def _create_retry_decorator(
    llm: ChatXinference,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    return create_base_retry_decorator(
        error_types=[Exception],
        max_retries=llm.model_kwargs.get("max_retries", 1),
        run_manager=run_manager,
    )


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


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        # TODO(codingl2k1): support function call
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


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role") or ""
    content = _dict.get("content") or ""
    # TODO(codingl2k1): support function call
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


def _openai_kwargs_to_xinference_kwargs(**kwargs: Any) -> Dict:
    messages = kwargs.pop("messages")
    prompt_message = messages.pop()
    return {
        "prompt": prompt_message["content"],
        "chat_history": messages,
        "generate_config": kwargs.pop("generate_config"),
    }


class ChatXinference(BaseChatModel):
    """`Xinference` Chat large language models API.

    To use, you should have the ``xinference-client`` python package installed.

    Any parameters that are valid to be passed to the model.chat call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatXinference
            client = ChatXinference(model_name="llama-2-chat")
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    client: Any = None  #: :meta private:
    server_url: str
    """URL of the xinference server"""
    model_uid: str
    """UID of the launched model"""
    model_kwargs: Dict[str, Any]
    """Holds any model parameters valid for `create` call not explicitly specified."""

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
        server_url = values.get("server_url")
        if not server_url:
            raise ValueError("Please specify the Xinference endpoint URL.")
        model_uid = values.get("model_uid")
        if not model_uid:
            raise ValueError("Please specify the Xinference model UID.")
        if values.get("model_kwargs", {}).get("n"):
            raise ValueError("n is not supported by Xinference.")
        if values.get("model_kwargs", {}).get("streaming"):
            raise ValueError("Please use stream instead of streaming.")

        try:
            from xinference_client import RESTfulClient as Client

            client = Client(server_url)
            values["client"] = client.get_model(model_uid=model_uid)
        except ImportError:
            raise ValueError(
                "Could not import xinference_client python package. "
                "Please install it with `pip install xinference-client`."
            )
        except Exception as e:
            raise ValueError(
                f"Could not connect to Xinference, "
                f"endpoint: {server_url}, model: {model_uid}"
            ) from e
        return values

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            kwargs = _openai_kwargs_to_xinference_kwargs(**kwargs)
            return self.client.chat(**kwargs)

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            kwargs = _openai_kwargs_to_xinference_kwargs(**kwargs)
            maybe_sync_iter = await asyncio.to_thread(self.client.chat, **kwargs)
            if isinstance(maybe_sync_iter, types.GeneratorType):
                return _to_async_iter(maybe_sync_iter)
            else:
                return maybe_sync_iter

        return await _completion_with_retry(**kwargs)

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
        return {"token_usage": overall_token_usage, "model_name": self.model_uid}

    @staticmethod
    def _get_merge_kwargs(kwargs1: Dict, kwargs2: Dict) -> Dict:
        generate_config1 = kwargs1.get("generate_config", {})
        generate_config2 = kwargs2.get("generate_config", {})
        return {
            **kwargs1,
            **kwargs2,
            "generate_config": {**generate_config1, **generate_config2},
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        merged_kwargs = self._get_merge_kwargs(
            {"generate_config": self.model_kwargs}, kwargs
        )
        generate_config = {**merged_kwargs.get("generate_config", {}), "stream": True}
        message_dicts, params = self._create_message_dicts(
            messages, stop, generate_config
        )
        params = self._get_merge_kwargs(params, kwargs)

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts,
            run_manager=run_manager,
            **params,
        ):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=chunk)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        merged_kwargs = self._get_merge_kwargs(
            {"generate_config": self.model_kwargs}, kwargs
        )
        generate_config = merged_kwargs.get("generate_config", {})
        should_stream = generate_config and generate_config.get("stream")
        if should_stream:
            stream_iter = self._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(
            messages, stop, generate_config
        )
        params = self._get_merge_kwargs(params, kwargs)
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_message_dicts(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        generate_config: Optional[Dict],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if stop is not None and generate_config is not None:
            if "stop" in generate_config:
                raise ValueError("`stop` found in both the input and default params.")
            generate_config["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, {"generate_config": generate_config}

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_uid}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        merged_kwargs = self._get_merge_kwargs(
            {"generate_config": self.model_kwargs}, kwargs
        )
        generate_config = {**merged_kwargs.get("generate_config", {}), "stream": True}
        message_dicts, params = self._create_message_dicts(
            messages, stop, generate_config
        )
        params = self._get_merge_kwargs(params, kwargs)

        default_chunk_class = AIMessageChunk
        async for chunk in await self.acompletion_with_retry(
            messages=message_dicts,
            run_manager=run_manager,
            **params,
        ):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                await run_manager.on_llm_new_token(token=chunk.content, chunk=chunk)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        merged_kwargs = self._get_merge_kwargs(
            {"generate_config": self.model_kwargs}, kwargs
        )
        generate_config = merged_kwargs.get("generate_config", {})
        should_stream = generate_config and generate_config.get("stream")
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(
            messages, stop, generate_config
        )
        params = self._get_merge_kwargs(params, kwargs)
        response = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"server_url": self.server_url},
            **{"model_uid": self.model_uid},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "xinference-chat"
