from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM, create_base_retry_decorator
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ErnieBot(LLM):
    """ERNIE Bot large language models.

    To use, you should have the ``erniebot`` python package installed, and the
    environment variable ``EB_ACCESS_TOKEN`` set with your AI Studio access
    token.

    Example:
        .. code-block:: python

            from langchain.llms import ErnieBot
            erniebot = ErnieBot(model="ernie-bot")
    """

    client: Any = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    aistudio_access_token: Optional[str] = None
    """AI Studio access token."""
    streaming: Optional[bool] = False
    """Whether to stream the results or not."""
    model: str = "ernie-bot"
    """Model to use."""
    top_p: Optional[float] = 0.8
    """Parameter of nucleus sampling that affects the diversity of generated content."""
    temperature: Optional[float] = 0.95
    """Sampling temperature to use."""
    penalty_score: Optional[float] = 1
    """Penalty assigned to tokens that have been generated."""
    request_timeout: Optional[int] = 60
    """How many seconds to wait for the server to send data before giving up."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ERNIE Bot API."""
        normal_params = {
            "model": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
            "request_timeout": self.request_timeout,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self._default_params

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        auth_cfg: Dict[str, Optional[str]] = {
            "api_type": "aistudio",
            "access_token": self.aistudio_access_token,
        }
        return {**{"_config_": auth_cfg}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "erniebot"

    @root_validator()
    def validate_enviroment(cls, values: Dict) -> Dict:
        values["aistudio_access_token"] = get_from_dict_or_env(
            values,
            "aistudio_access_token",
            "EB_ACCESS_TOKEN",
        )
        try:
            import erniebot

            values["client"] = erniebot.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import erniebot python package. "
                "Please install it with `pip install erniebot`."
            )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            text = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                text += chunk.text
            return text
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = [self._build_user_message_from_prompt(prompt)]
            params["stream"] = False
            response = _create_completion_with_retry(
                self, run_manager=run_manager, **params
            )
            text = response["result"]
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            text = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                text += chunk.text
            return text
        else:
            params = self._invocation_params
            params.update(kwargs)
            params["messages"] = [self._build_user_message_from_prompt(prompt)]
            params["stream"] = False
            response = await _acreate_completion_with_retry(
                self, run_manager=run_manager, **params
            )
            text = response["result"]
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if stop is not None:
            raise TypeError(
                "Currently, `stop` is not supported when streaming is enabled."
            )
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = [self._build_user_message_from_prompt(prompt)]
        params["stream"] = True
        for resp in _create_completion_with_retry(
            self, run_manager=run_manager, **params
        ):
            chunk = self._build_chunk_from_response(resp)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        if stop is not None:
            raise TypeError(
                "Currently, `stop` is not supported when streaming is enabled."
            )
        params = self._invocation_params
        params.update(kwargs)
        params["messages"] = [self._build_user_message_from_prompt(prompt)]
        params["stream"] = True
        async for resp in await _acreate_completion_with_retry(
            self, run_manager=run_manager, **params
        ):
            chunk = self._build_chunk_from_response(resp)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _build_chunk_from_response(
        self, response: Mapping[str, Any]
    ) -> GenerationChunk:
        return GenerationChunk(text=response["result"])

    def _build_user_message_from_prompt(self, prompt: str) -> Dict[str, str]:
        return {"role": "user", "content": prompt}


def _create_completion_with_retry(
    llm: ErnieBot,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    def _client_create(**kwargs: Any) -> Any:
        return llm.client.create(**kwargs)

    return _client_create(**kwargs)


async def _acreate_completion_with_retry(
    llm: ErnieBot,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _client_acreate(**kwargs: Any) -> Any:
        return await llm.client.acreate(**kwargs)

    return await _client_acreate(**kwargs)


def _create_retry_decorator(
    llm: ErnieBot,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    import erniebot

    errors: List[Type[BaseException]] = [
        erniebot.errors.TimeoutError,
        erniebot.errors.RequestLimitError,
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
