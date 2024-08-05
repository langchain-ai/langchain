"""MindsDB Endpoint embeddings wrapper. Relies heavily on OpenAIEmbeddings."""

from __future__ import annotations

from typing import Text, Dict

from langchain_community.utils.openai import is_openai_v1
from langchain_community.embeddings.openai import OpenAIEmbeddings

from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

DEFAULT_API_BASE = "https://llm.mdb.ai"
DEFAULT_MODEL = "text-embedding-ada-002"
EMBEDDING_MODELS = ["text-embedding-ada-002"]


class AIMindEmbeddings(OpenAIEmbeddings):
    """
    `Minds Endpoint` large language models for embeddings from MindsDB.

    See https://docs.mdb.ai/ for information about MindsDB and the MindsDB Endpoint.

    To use this chat model, you should have the ``openai`` python package installed, and the environment variable ``MINDSDB_API_KEY`` set with your API key.
    Alternatively, you can use the mindsdb_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.embeddings.create` call can be passed in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import AIMindEmbeddings
            embeddings = AIMindEmbeddings(model="text-embedding-ada-002")
    """

    @property
    def _llm_type(self) -> Text:
        """Return type of chat model."""
        return "ai-mind-embeddings"

    @property
    def lc_secrets(self) -> Dict[Text, Text]:
        return {"mindsdb_api_key": "MINDSDB_API_KEY"}
    
    mindsdb_api_key: SecretStr = Field(default=None)
    mindsdb_api_base: str = Field(default=DEFAULT_API_BASE)
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate that the Minds Endpoint API credentials are provided and create an OpenAI client based on the version of the `openai` package that is being used.
        Further, validate that the chosen model is supported by the MindsDB API.

        Args:
            values: The values passed to the class constructor.
        """
        # Validate that the API key and base URL are available.
        values["mindsdb_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "mindsdb_api_key",
                "MINDSDB_API_KEY",
            )
        )
        values["mindsdb_api_base"] = get_from_dict_or_env(
            values,
            "mindsdb_api_base",
            "MINDSDB_API_BASE",
            default=DEFAULT_API_BASE,
        )

        # Validate that the `openai` package can be imported.
        try:
            import openai

        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e

        # Set the client based on the version of the `openai` package that is being used.
        if is_openai_v1():
            client_params = {
                "api_key": values["mindsdb_api_key"].get_secret_value(),
                "base_url": values["mindsdb_api_base"],
            }
            values["client"] = openai.OpenAI(**client_params).embeddings
        else:
            values["openai_api_base"] = values["mindsdb_api_base"]
            values["openai_api_key"] = values["mindsdb_api_key"].get_secret_value()
            values["client"] = openai.Embedding

        # Validate that the chosen embeddings model provided is supported.
        model_name = values["model_name"]       
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Model name {model_name} not found in available models: "
                f"{EMBEDDING_MODELS}.",
            )

        return values