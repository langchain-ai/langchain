"""Anyscale embeddings wrapper."""

from __future__ import annotations

from typing import Dict, Optional

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import Field, SecretStr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL = "thenlper/gte-large"


class AnyscaleEmbeddings(OpenAIEmbeddings):
    """`Anyscale` Embeddings API."""

    anyscale_api_key: Optional[SecretStr] = Field(default=None)
    """AnyScale Endpoints API keys."""
    model: str = Field(default=DEFAULT_MODEL)
    """Model name to use."""
    anyscale_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests."""
    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations of the embeddings API"""
    embedding_ctx_length: int = 500
    """The maximum number of tokens to embed at once."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "anyscale_api_key": "ANYSCALE_API_KEY",
        }

    @pre_init
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
        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if is_openai_v1():
            # For backwards compatibility.
            client_params = {
                "api_key": values["anyscale_api_key"].get_secret_value(),
                "base_url": values["anyscale_api_base"],
            }
            values["client"] = openai.OpenAI(**client_params).embeddings
        else:
            values["openai_api_base"] = values["anyscale_api_base"]
            values["openai_api_key"] = values["anyscale_api_key"].get_secret_value()
            values["client"] = openai.Embedding
        return values

    @property
    def _llm_type(self) -> str:
        return "anyscale-embedding"
