"""Wrapper around Together AI's Chat Completions API."""

import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_openai.chat_models.base import BaseChatOpenAI


class ChatTogether(BaseChatOpenAI):
    """ChatTogether chat model.

    To use, you should have the environment variable `TOGETHER_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_together import ChatTogether


            model = ChatTogether()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"together_api_key": "TOGETHER_API_KEY"}
        """
        return {"together_api_key": "TOGETHER_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "together"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.together_api_base:
            attributes["together_api_base"] = self.together_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "together-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "together"
        return params

    model_name: str = Field(default="meta-llama/Llama-3-8b-chat-hf", alias="model")
    """Model name to use."""
    together_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `TOGETHER_API_KEY` if not provided."""
    together_api_base: Optional[str] = Field(
        default="https://api.together.ai/v1/", alias="base_url"
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["together_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "together_api_key", "TOGETHER_API_KEY")
        )
        values["together_api_base"] = values["together_api_base"] or os.getenv(
            "TOGETHER_API_BASE"
        )

        client_params = {
            "api_key": (
                values["together_api_key"].get_secret_value()
                if values["together_api_key"]
                else None
            ),
            "base_url": values["together_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }

        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return values
