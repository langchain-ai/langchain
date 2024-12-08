from typing import Dict

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import Field, SecretStr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"
DEFAULT_MODEL = "BAAI/bge-en-icl"


class NebiusAIStudioEmbeddings(OpenAIEmbeddings):
    """Nebius AI Studio embedding models.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``NEBIUS_API_KEY`` set with your API key.
    Alternatively, you can use the nebius_api_key keyword argument.
    """

    nebius_api_key: SecretStr = Field(default=None)
    """Nebius AI Studio API keys."""
    endpoint_url: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests."""
    model: str = Field(default=DEFAULT_MODEL)
    """Model name to use."""
    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations of the embeddings API"""

    @property
    def _llm_type(self) -> str:
        """Return type of embeddings model."""
        return "nebius-ai-studio-embeddings"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"nebius_api_token": "NEBIUS_API_KEY"}

    @pre_init
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        values["endpoint_url"] = get_from_dict_or_env(
            values,
            "endpoint_url",
            "ENDPOINT_URL",
            default=DEFAULT_API_BASE,
        )
        values["nebius_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "nebius_api_key", "NEBIUS_API_KEY")
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
                    "api_key": values["nebius_api_key"].get_secret_value(),
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
                values["openai_api_key"] = values["nebius_api_key"].get_secret_value()
                values["client"] = openai.Embedding  # type: ignore[attr-defined]
                values["async_client"] = openai.Embedding  # type: ignore[attr-defined]

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values
