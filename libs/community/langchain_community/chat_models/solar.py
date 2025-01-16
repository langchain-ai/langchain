"""Wrapper around Solar chat models."""

from typing import Dict

from langchain_core._api import deprecated
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import ConfigDict, Field

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.solar import SOLAR_SERVICE_URL_BASE, SolarCommon


@deprecated(  # type: ignore[arg-type]
    since="0.0.34", removal="1.0", alternative_import="langchain_upstage.ChatUpstage"
)
class SolarChat(SolarCommon, ChatOpenAI):  # type: ignore[override, override]
    """Wrapper around Solar large language models.
    To use, you should have the ``openai`` python package installed, and the
    environment variable ``SOLAR_API_KEY`` set with your API key.
    (Solar's chat API is compatible with OpenAI's SDK.)
    Referenced from https://console.upstage.ai/services/solar
    Example:
        .. code-block:: python

            from langchain_community.chat_models.solar import SolarChat

            solar = SolarChat(model="solar-mini")
    """

    max_tokens: int = Field(default=1024)

    # this is needed to match ChatOpenAI superclass
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the environment is set up correctly."""
        values["solar_api_key"] = get_from_dict_or_env(
            values, "solar_api_key", "SOLAR_API_KEY"
        )

        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        client_params = {
            "api_key": values["solar_api_key"],
            "base_url": (
                values["base_url"] if "base_url" in values else SOLAR_SERVICE_URL_BASE
            ),
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values
