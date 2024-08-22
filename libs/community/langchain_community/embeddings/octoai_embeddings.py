from typing import Dict

from langchain_core.pydantic_v1 import Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://text.octoai.run/v1/"
DEFAULT_MODEL = "thenlper/gte-large"


class OctoAIEmbeddings(OpenAIEmbeddings):
    """OctoAI Compute Service embedding models.

    See https://octo.ai/ for information about OctoAI.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``OCTOAI_API_TOKEN`` set with your API token.
    Alternatively, you can use the octoai_api_token keyword argument.
    """

    octoai_api_token: SecretStr = Field(default=None)
    """OctoAI Endpoints API keys."""
    endpoint_url: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests."""
    model: str = Field(default=DEFAULT_MODEL)
    """Model name to use."""
    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations of the embeddings API"""

    @property
    def _llm_type(self) -> str:
        """Return type of embeddings model."""
        return "octoai-embeddings"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"octoai_api_token": "OCTOAI_API_TOKEN"}

    @pre_init
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        values["endpoint_url"] = get_from_dict_or_env(
            values,
            "endpoint_url",
            "ENDPOINT_URL",
            default=DEFAULT_API_BASE,
        )
        values["octoai_api_token"] = convert_to_secret_str(
            get_from_dict_or_env(values, "octoai_api_token", "OCTOAI_API_TOKEN")
        )
        values["model"] = get_from_dict_or_env(
            values,
            "model",
            "MODEL",
            default=DEFAULT_MODEL,
        )

        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["octoai_api_token"].get_secret_value(),
                    "base_url": values["endpoint_url"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).embeddings
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).embeddings
            else:
                values["openai_api_base"] = values["endpoint_url"]
                values["openai_api_key"] = values["octoai_api_token"].get_secret_value()
                values["client"] = openai.Embedding
                values["async_client"] = openai.Embedding

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values
