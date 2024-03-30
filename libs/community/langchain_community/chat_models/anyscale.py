"""Anyscale Endpoints chat wrapper. Relies heavily on ChatOpenAI."""
from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Dict, Optional, Set

import requests
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_community.chat_models.openai import (
    ChatOpenAI,
    _import_tiktoken,
)
from langchain_community.utils.openai import is_openai_v1

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


class ChatAnyscale(ChatOpenAI):
    """`Anyscale` Chat large language models.

    See https://www.anyscale.com/ for information about Anyscale.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``ANYSCALE_API_KEY`` set with your API key.
    Alternatively, you can use the anyscale_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatAnyscale
            chat = ChatAnyscale(model_name="meta-llama/Llama-2-7b-chat-hf")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anyscale-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"anyscale_api_key": "ANYSCALE_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    anyscale_api_key: SecretStr = Field(default=None)
    """AnyScale Endpoints API keys."""
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    anyscale_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests,
    leave blank if not using a proxy or service emulator."""
    anyscale_proxy: Optional[str] = None
    """To support explicit proxy for Anyscale."""
    available_models: Optional[Set[str]] = None
    """Available models from Anyscale API."""

    @staticmethod
    def get_available_models(
        anyscale_api_key: Optional[str] = None,
        anyscale_api_base: str = DEFAULT_API_BASE,
    ) -> Set[str]:
        """Get available models from Anyscale API."""
        try:
            anyscale_api_key = anyscale_api_key or os.environ["ANYSCALE_API_KEY"]
        except KeyError as e:
            raise ValueError(
                "Anyscale API key must be passed as keyword argument or "
                "set in environment variable ANYSCALE_API_KEY.",
            ) from e

        models_url = f"{anyscale_api_base}/models"
        models_response = requests.get(
            models_url,
            headers={
                "Authorization": f"Bearer {anyscale_api_key}",
            },
        )

        if models_response.status_code != 200:
            raise ValueError(
                f"Error getting models from {models_url}: "
                f"{models_response.status_code}",
            )

        return {model["id"] for model in models_response.json()["data"]}

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        values["anyscale_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "anyscale_api_key",
                "ANYSCALE_API_KEY",
            )
        )
        values["anyscale_api_base"] = get_from_dict_or_env(
            values,
            "anyscale_api_base",
            "ANYSCALE_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "anyscale_proxy",
            "ANYSCALE_PROXY",
            default="",
        )
        try:
            import openai

        except ImportError as e:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e
        try:
            if is_openai_v1():
                client_params = {
                    "api_key": values["anyscale_api_key"].get_secret_value(),
                    "base_url": values["anyscale_api_base"],
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
                values["openai_api_base"] = values["anyscale_api_base"]
                values["openai_api_key"] = values["anyscale_api_key"].get_secret_value()
                values["client"] = openai.ChatCompletion
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
            values["anyscale_api_key"].get_secret_value(),
            values["anyscale_api_base"],
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

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        """Calculate num tokens with tiktoken package.
        Official documentation: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        """
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
