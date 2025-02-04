"""AiBrary Endpoints chat wrapper. Relies heavily on ChatOpenAI."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

import requests
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr, model_validator

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_community.chat_models.openai import (
    ChatOpenAI,
    _import_tiktoken,
)
from langchain_community.utils.openai import is_openai_v1

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.aibrary.dev/v0"
DEFAULT_MODEL = "gpt-4o"


class ChatAiBrary(ChatOpenAI):
    """`AiBrary` Chat large language models.

    See https://www.aibrary.dev/ for information about AiBrary.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``AIBRARY_API_KEY`` set with your API key.
    Alternatively, you can use the aibrary_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatAiBrary
            chat = ChatAiBrary(model_name="gpt-4o")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "aibrary-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"aibrary_api_key": "AIBRARY_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    aibrary_api_key: SecretStr = Field(default=SecretStr(""))
    """AiBrary AI API key. Get it here: https://www.aibrary.dev/dashboard/apikey"""
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    aibrary_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests,
    leave blank if not using a proxy or service emulator."""
    aibrary_proxy: Optional[str] = None
    """To support explicit proxy for AiBrary."""
    available_models: Optional[Set[str]] = None
    """Available models from AiBrary API."""

    @staticmethod
    def get_available_models(
        aibrary_api_key: Optional[str] = None,
        aibrary_api_base: str = DEFAULT_API_BASE,
    ) -> Set[str]:
        """Get available models from AiBrary API."""
        try:
            aibrary_api_key = aibrary_api_key or os.environ["AIBRARY_API_KEY"]
        except KeyError as e:
            raise ValueError(
                "AiBrary API key must be passed as keyword argument or "
                "set in environment variable AIBRARY_API_KEY.",
            ) from e

        models_url = f"{aibrary_api_base}/models"
        models_response = requests.get(
            models_url,
            headers={
                "Authorization": f"Bearer {aibrary_api_key}",
            },
        )

        if models_response.status_code != 200:
            raise ValueError(
                f"Error getting models from {models_url}: "
                f"{models_response.status_code}",
            )

        return {model["id"] for model in models_response.json()["data"]}

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate that api key and python package exists in environment."""
        values["aibrary_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "aibrary_api_key",
                "AIBRARY_API_KEY",
            )
        )
        values["aibrary_api_base"] = get_from_dict_or_env(
            values,
            "aibrary_api_base",
            "AIBRARY_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "aibrary_proxy",
            "AIBRARY_PROXY",
            default="",
        )
        try:
            import openai

        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e
        try:
            if is_openai_v1():
                client_params = {
                    "api_key": values["aibrary_api_key"].get_secret_value(),
                    "base_url": values["aibrary_api_base"],
                    # To do: future support
                    # "organization": values["openai_organization"],
                    # "timeout": values["request_timeout"],
                    # "max_retries": values["max_retries"],
                    # "default_headers": values["default_headers"],
                    # "default_query": values["default_query"],
                    # "http_client": values["http_client"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["aibrary_api_base"]
                values["openai_api_key"] = values["aibrary_api_key"].get_secret_value()
                values["client"] = openai.ChatCompletion  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`.",
            ) from exc

        if "model_name" not in values.keys():
            values["model_name"] = DEFAULT_MODEL

        model_name = values["model_name"]
        available_models = cls.get_available_models(
            values["aibrary_api_key"].get_secret_value(),
            values["aibrary_api_base"],
        )

        if model_name not in available_models:
            raise ValueError(
                f"Model name {model_name} not found in available models: "
                f"{available_models}.",
            )

        values["available_models"] = available_models

        return values

    def _get_encoding_model(self) -> tuple[str, tiktoken.Encoding]:
        tiktoken_ = _import_tiktoken()
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken_.encoding_for_model("gpt-3.5-turbo-0301")
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Optional[
            Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]]
        ] = None,
    ) -> int:
        """Calculate num tokens with tiktoken package.
        Official documentation: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        """
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools."
            )
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        messages_dict = [convert_message_to_dict(m) for m in messages]
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
