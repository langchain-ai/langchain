"""KonkoAI chat wrapper."""

from __future__ import annotations

import logging
import os
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import requests
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import Field, SecretStr

from langchain_community.adapters.openai import (
    convert_message_to_dict,
)
from langchain_community.chat_models.openai import (
    ChatOpenAI,
    _convert_delta_to_message_chunk,
    generate_from_stream,
)
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://api.konko.ai/v1"
DEFAULT_MODEL = "meta-llama/Llama-2-13b-chat-hf"

logger = logging.getLogger(__name__)


class ChatKonko(ChatOpenAI):  # type: ignore[override]
    """`ChatKonko` Chat large language models API.

    To use, you should have the ``konko`` python package installed, and the
    environment variable ``KONKO_API_KEY`` and ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the konko.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatKonko
            llm = ChatKonko(model="meta-llama/Llama-2-13b-chat-hf")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"konko_api_key": "KONKO_API_KEY", "openai_api_key": "OPENAI_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    client: Any = None  #: :meta private:
    model: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    konko_api_key: Optional[str] = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: int = 20
    """Maximum number of tokens to generate."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["konko_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "konko_api_key", "KONKO_API_KEY")
        )
        try:
            import konko

        except ImportError:
            raise ImportError(
                "Could not import konko python package. "
                "Please install it with `pip install konko`."
            )
        try:
            if is_openai_v1():
                values["client"] = konko.chat.completions
            else:
                values["client"] = konko.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`konko` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the konko package. Try upgrading it "
                "with `pip install --upgrade konko`."
            )

        if not hasattr(konko, "_is_legacy_openai"):
            warnings.warn(
                "You are using an older version of the 'konko' package. "
                "Please consider upgrading to access new features."
            )

        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Konko API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    @staticmethod
    def get_available_models(
        konko_api_key: Union[str, SecretStr, None] = None,
        openai_api_key: Union[str, SecretStr, None] = None,
        konko_api_base: str = DEFAULT_API_BASE,
    ) -> Set[str]:
        """Get available models from Konko API."""

        # Try to retrieve the OpenAI API key if it's not passed as an argument
        if not openai_api_key:
            try:
                openai_api_key = convert_to_secret_str(os.environ["OPENAI_API_KEY"])
            except KeyError:
                pass  # It's okay if it's not set, we just won't use it
        elif isinstance(openai_api_key, str):
            openai_api_key = convert_to_secret_str(openai_api_key)

        # Try to retrieve the Konko API key if it's not passed as an argument
        if not konko_api_key:
            try:
                konko_api_key = convert_to_secret_str(os.environ["KONKO_API_KEY"])
            except KeyError:
                raise ValueError(
                    "Konko API key must be passed as keyword argument or "
                    "set in environment variable KONKO_API_KEY."
                )
        elif isinstance(konko_api_key, str):
            konko_api_key = convert_to_secret_str(konko_api_key)

        models_url = f"{konko_api_base}/models"

        headers = {
            "Authorization": f"Bearer {konko_api_key.get_secret_value()}",
        }

        if openai_api_key:
            headers["X-OpenAI-Api-Key"] = cast(
                SecretStr, openai_api_key
            ).get_secret_value()

        models_response = requests.get(models_url, headers=headers)

        if models_response.status_code != 200:
            raise ValueError(
                f"Error getting models from {models_url}: {models_response.status_code}"
            )

        return {model["id"] for model in models_response.json()["data"]}

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

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
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

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
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model}, **self._default_params}

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the konko client."""
        return {**self._default_params}

    def _get_invocation_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {
            "model": self.model,
            **super()._get_invocation_params(stop=stop),
            **self._default_params,
            **kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "konko-chat"
