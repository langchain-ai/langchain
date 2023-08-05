"""Anyscale Endpoints chat wrapper. Relies heavily on ChatOpenAI."""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from pydantic import Field, root_validator

from langchain.chat_models.openai import (
    ChatOpenAI,
    _convert_message_to_dict,
    _import_tiktoken,
)
from langchain.schema.messages import BaseMessage
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


DEFAULT_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


class ChatAnyscale(ChatOpenAI):
    """Wrapper around Anyscale Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``ANYSCALE_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatAnyscale
            chat = ChatAnyscale(model_name="meta-llama/Llama-2-7b-chat-hf")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anyscale-chat"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"anyscale_api_key": "ANYSCALE_API_KEY"}

    anyscale_api_key: str | None = None
    """AnyScale Endpoints API keys."""
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    anyscale_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests,
    leave blank if not using a proxy or service emulator."""
    # to support explicit proxy for Anyscale
    anyscale_proxy: str | None = None
    available_models: set[str] | None = None

    @root_validator()
    def validate_environment_override(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values,
            "anyscale_api_key",
            "ANYSCALE_API_KEY",
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "anyscale_api_base",
            "ANYSCALE_API_BASE",
            default="https://api.endpoints.anyscale.com",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "anyscale_proxy",
            "anyscale_proxy",
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
            values["client"] = openai.ChatCompletion
        except AttributeError as exc:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`.",
            ) from exc
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        model_name = values["model_name"]
        available_models: set[str] = {
            model["id"] for model in openai.Model.list()["data"]
        }
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

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        tokens_per_message = 3
        tokens_per_name = 1
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
