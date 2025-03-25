import json
from typing import Any, Dict, List, Optional

from langchain_core._api import deprecated
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
VALID_TASKS = ("feature-extraction",)


@deprecated(
    since="0.2.2",
    removal="1.0",
    alternative_import="langchain_huggingface.HuggingFaceEndpointEmbeddings",
)
class HuggingFaceHubEmbeddings(BaseModel, Embeddings):
    """HuggingFaceHub embedding models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceHubEmbeddings
            model = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceHubEmbeddings(
                model=model,
                task="feature-extraction",
                huggingfacehub_api_token="my-api-key",
            )
    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: Optional[str] = None
    """Model name to use."""
    repo_id: Optional[str] = None
    """Huggingfacehub repository id, for backward compatibility."""
    task: Optional[str] = "feature-extraction"
    """Task to call the model with."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""

    huggingfacehub_api_token: Optional[str] = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )

        try:
            from huggingface_hub import AsyncInferenceClient, InferenceClient

            if values.get("model"):
                values["repo_id"] = values["model"]
            elif values.get("repo_id"):
                values["model"] = values["repo_id"]
            else:
                values["model"] = DEFAULT_MODEL
                values["repo_id"] = DEFAULT_MODEL

            client = InferenceClient(
                model=values["model"],
                token=huggingfacehub_api_token,
            )

            async_client = AsyncInferenceClient(
                model=values["model"],
                token=huggingfacehub_api_token,
            )

            values["client"] = client
            values["async_client"] = async_client

        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        return values

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Post init validation for the class."""
        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {self.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        responses = self.client.post(
            json={"inputs": texts, **_model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = await self.async_client.post(
            json={"inputs": texts, "parameters": _model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    def embed_query(self, text: str) -> List[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> List[float]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = (await self.aembed_documents([text]))[0]
        return response
