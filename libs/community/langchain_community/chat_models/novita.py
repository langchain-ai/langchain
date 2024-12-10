"""Wrapper around Novita chat models."""

from typing import Any, Dict

from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from pydantic import Field, SecretStr, model_validator

from langchain_community.chat_models import ChatOpenAI

NOVITA_API_BASE = "https://api.novita.ai/v3/openai"


class ChatNovita(ChatOpenAI):  # type: ignore[misc]
    """Novita AI LLM.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``NOVITA_API_KEY`` set with your API key.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatNovita

            chat = ChatNovita(model="gryphe/mythomax-l2-13b")
    """

    novita_api_key: SecretStr = Field(default=None, alias="api_key")
    model_name: str = Field(default="gryphe/mythomax-l2-13b", alias="model")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the environment is set up correctly."""
        values["novita_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["novita_api_key", "api_key", "openai_api_key"],
                "NOVITA_API_KEY",
            )
        )

        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        client_params = {
            "api_key": values["novita_api_key"].get_secret_value(),
            "base_url": values.get("base_url", NOVITA_API_BASE),
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values
