import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import openai
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_openai import ChatOpenAI


class ChatUpstage(ChatOpenAI):
    """ChatUpstage chat model.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import ChatUpstage


            model = ChatUpstage()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"upstage_api_key": "UPSTAGE_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "upstage"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.upstage_api_base:
            attributes["upstage_api_base"] = self.upstage_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "upstage-chat"

    model_name: str = Field(default="solar-1-mini-chat", alias="model")
    """Model name to use."""
    upstage_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `UPSTAGE_API_KEY` if not provided."""
    upstage_api_base: Optional[str] = Field(
        default="https://api.upstage.ai/v1/solar", alias="base_url"
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["upstage_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "upstage_api_key", "UPSTAGE_API_KEY")
        )
        values["upstage_api_base"] = values["upstage_api_base"] or os.getenv(
            "UPSTAGE_API_BASE"
        )

        client_params = {
            "api_key": (
                values["upstage_api_key"].get_secret_value()
                if values["upstage_api_key"]
                else None
            ),
            "base_url": values["upstage_api_base"],
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
