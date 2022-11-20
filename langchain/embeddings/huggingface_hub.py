"""Wrapper around HuggingFace Hub embedding models."""
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings

DEFAULT_REPO_ID = "sentence-transformers/all-mpnet-base-v2"


class HuggingFaceHubEmbeddings(BaseModel, Embeddings):
    """Wrapper around HuggingFaceHub embedding models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceHubEmbeddings
            repo_id = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceHubEmbeddings(
                repo_id=repo_id, huggingfacehub_api_token="my-api-key"
            )
    """

    client: Any  #: :meta private:
    repo_id: str = DEFAULT_REPO_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    huggingfacehub_api_token: Optional[str] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = values.get("huggingfacehub_api_token")
        if huggingfacehub_api_token is None or huggingfacehub_api_token == "":
            raise ValueError(
                "Did not find HuggingFace API token, please add an environment variable"
                " `HUGGINGFACEHUB_API_TOKEN` which contains it, or pass"
                " `huggingfacehub_api_token` as a named parameter."
            )
        try:
            from huggingface_hub.inference_api import InferenceApi

            repo_id = values.get("repo_id", DEFAULT_REPO_ID)
            client = InferenceApi(
                repo_id=repo_id,
                token=huggingfacehub_api_token,
                task="feature-extraction",
            )
            values["client"] = client
        except ImportError:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please it install it with `pip install huggingface_hub`."
            )
        return values

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
        responses = self.client(inputs=texts, params=_model_kwargs)
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response
