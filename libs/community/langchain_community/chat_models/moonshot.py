"""Wrapper around Moonshot chat models."""
from typing import Dict

from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.moonshot import MOONSHOT_SERVICE_URL_BASE, MoonshotCommon


class MoonshotChat(MoonshotCommon, ChatOpenAI):
    """Wrapper around Moonshot large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``MOONSHOT_API_KEY`` set with your API key.
    (Moonshot's chat API is compatible with OpenAI's SDK.)

    Referenced from https://platform.moonshot.cn/docs

    Example:
        .. code-block:: python

            from langchain_community.chat_models.moonshot import MoonshotChat

            moonshot = MoonshotChat(model="moonshot-v1-8k")
    """

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is set up correctly."""
        values["moonshot_api_key"] = get_from_dict_or_env(
            values, "moonshot_api_key", "MOONSHOT_API_KEY"
        )

        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        client_params = {
            "api_key": values["moonshot_api_key"],
            "base_url": values["base_url"]
            if "base_url" in values
            else MOONSHOT_SERVICE_URL_BASE,
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values
