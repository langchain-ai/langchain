from typing import Any, Dict, List, Optional
from pydantic import BaseModel, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env

from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticRepresentation,
    SemanticEmbeddingRequest,
)


class AlephAlphaAsymmetricSemanticEmbedding(BaseModel, Embeddings):
    """
    Wrapper around Aleph Alpha Asymmetric Embeddings
    AA provides you with an endpoint for to embed the Document and a Query.
    The models were optimized to make the embeddings of document and the query for
    a document as similar to each other as possible.

    To learn more check out: https://docs.aleph-alpha.com/docs/tasks/semantic_embed/
    """

    client: Any  #: :meta private:

    model: Optional[str] = "luminous-base"
    """Model name to use."""
    hosting: Optional[str] = ("https://api.aleph-alpha.com",)
    normalize: Optional[bool] = True
    compress_to_size: Optional[int] = 128
    contextual_control_threshold: Optional[int] = None
    control_log_additive: Optional[bool] = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        aleph_alpha_api_key = get_from_dict_or_env(
            values, "aleph_alpha_api_key", "ALEPH_ALPHA_API_KEY"
        )
        try:
            values["client"] = Client(token=aleph_alpha_api_key)
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please it install it with `pip install aleph_alpha_client`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Aleph Alpha's asymmetric Document endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        document_embeddings = []

        for text in texts:
            document_params = {
                "prompt": Prompt.from_text(text),
                "representation": SemanticRepresentation.Document,
                "compress_to_size": self.compress_to_size,
                "normalize": self.normalize,
                "contextual_control_threshold": self.contextual_control_threshold,
                "control_log_additive": self.control_log_additive,
            }

            document_request = SemanticEmbeddingRequest(**document_params)
            document_response = self.client.semantic_embed(
                request=document_request, model=self.model
            )

            document_embeddings.append(document_response.embedding)

        return document_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Aleph Alpha's asymmetric, query embedding endpoint
        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        symmetric_params = {
            "prompt": Prompt.from_text(text),
            "representation": SemanticRepresentation.Query,
            "compress_to_size": self.compress_to_size,
            "normalize": self.normalize,
            "contextual_control_threshold": self.contextual_control_threshold,
            "control_log_additive": self.control_log_additive,
        }

        symmetric_request = SemanticEmbeddingRequest(**symmetric_params)
        symmetric_response = self.client.semantic_embed(
            request=symmetric_request, model=self.model
        )

        return symmetric_response.embedding


class AlephAlphaSymmetricSemanticEmbedding(AlephAlphaAsymmetricSemanticEmbedding):
    # The symmetric version of the Aleph Alpha's embeddings. The main difference is that here, both the documents and
    # queries are embedded with a SemanticRepresentation.Symmetric
    def _embed(self, text: str) -> List[float]:
        query_params = {
            "prompt": Prompt.from_text(text),
            "representation": SemanticRepresentation.Symmetric,
            "compress_to_size": self.compress_to_size,
            "normalize": self.normalize,
            "contextual_control_threshold": self.contextual_control_threshold,
            "control_log_additive": self.control_log_additive,
        }

        query_request = SemanticEmbeddingRequest(**query_params)
        query_response = self.client.semantic_embed(
            request=query_request, model=self.model
        )

        return query_response.embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Aleph Alpha's Document endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        document_embeddings = []

        for text in texts:
            document_embeddings.append(self._embed(text))
        return document_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Aleph Alpha's asymmetric, query embedding endpoint
        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed(text)
