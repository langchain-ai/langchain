"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__file__)


def _create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: ChatOpenAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatOpenAI(BaseChatModel, BaseModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: int = 256
    """Maximum number of tokens to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        import openai

        min_seconds = 4
        max_seconds = 10
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            for stream_resp in self.completion_with_retry(
                messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                self.callback_manager.on_llm_new_token(
                    token,
                    verbose=self.verbose,
                )
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(messages=message_dicts, **params)
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        return ChatResult(generations=generations)

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(
                self, messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if self.callback_manager.is_async:
                    await self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
                else:
                    self.callback_manager.on_llm_new_token(
                        token,
                        verbose=self.verbose,
                    )
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role}
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            full_response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
            generations = []
            for res in full_response["choices"]:
                message = _convert_dict_to_message(res["message"])
                gen = ChatGeneration(message=message)
                generations.append(gen)
            return ChatResult(generations=generations)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.8 or below
        if sys.version_info[1] <= 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3.5-Turbo encoder instance
        enc = tiktoken.encoding_for_model(self.model_name)

        # encode the text using the GPT-3.5-Turbo encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)
