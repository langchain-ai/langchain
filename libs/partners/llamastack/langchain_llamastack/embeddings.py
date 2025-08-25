"""LangChain embeddings integration for Llama Stack."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env

try:
    from llama_stack_client import LlamaStackClient
except ImportError:
    raise ImportError(
        "llama-stack-client is required to use LlamaStackEmbeddings. "
        "Install it with `pip install llama-stack-client`"
    )

logger = logging.getLogger(__name__)


class LlamaStackEmbeddings(Embeddings):
    """LangChain embeddings for Llama Stack.

    This class provides a LangChain-compatible interface for Llama Stack's
    embedding models through the embeddings endpoint.

    Example:
        .. code-block:: python

            from langchain_llamastack import LlamaStackEmbeddings

            # Initialize with default settings
            embeddings = LlamaStackEmbeddings(
                model="all-minilm",
                base_url="http://localhost:8321"
            )

            # Embed a single document
            embedding = embeddings.embed_query("Hello, world!")
            print(f"Embedding dimension: {len(embedding)}")

            # Embed multiple documents
            docs = ["Hello", "World", "AI is amazing"]
            embeddings_list = embeddings.embed_documents(docs)
            print(f"Embedded {len(embeddings_list)} documents")
    """

    def __init__(
        self,
        model: str = "all-minilm",
        base_url: str = "http://localhost:8321",
        chunk_size: int = 1000,
        max_retries: int = 3,
        request_timeout: float = 30.0,
        **kwargs: Any,
    ):
        """Initialize LlamaStackEmbeddings.

        Args:
            model: Model name to use for embeddings
            base_url: Base URL for the Llama Stack server
            chunk_size: Maximum number of texts to embed in each batch
            max_retries: Maximum number of retries for API calls
            request_timeout: Request timeout in seconds
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        self.model = model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Setup the client
        self._setup_client()

    def _setup_client(self):
        """Setup the Llama Stack client."""
        # Initialize the client
        try:
            client_kwargs = {"base_url": self.base_url}
            self.client = LlamaStackClient(**client_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to initialize Llama Stack client: {e}")

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "llamastack-embeddings"

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return self._embed_texts(texts)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to embed texts after {self.max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
        return []

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using LlamaStack OpenAI-compatible endpoint."""
        try:
            import httpx

            logger.info(f"Embedding {len(texts)} texts with model {self.model}")

            # Use OpenAI-compatible embeddings endpoint
            api_url = f"{self.base_url}/v1/openai/v1/embeddings"

            # Prepare headers
            headers = {"Content-Type": "application/json"}

            # Prepare request payload in OpenAI format
            payload = {"model": self.model, "input": texts}

            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            # Extract embeddings from OpenAI-format response
            embeddings = []
            if "data" in result:
                for item in result["data"]:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
                    else:
                        raise ValueError(f"No embedding found in response item: {item}")
            else:
                raise ValueError(f"No data in embeddings response: {result}")

            logger.info(
                f"Successfully generated {len(embeddings)} embeddings via LlamaStack OpenAI endpoint"
            )
            return embeddings

        except Exception as e:
            logger.error(f"LlamaStack embeddings failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of documents to embed.

        Returns:
            List of embeddings, one for each document.
        """
        # Process texts in chunks to avoid overwhelming the API
        all_embeddings = []

        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            chunk_embeddings = self._embed_with_retry(chunk)
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        embeddings = self._embed_with_retry([text])
        return embeddings[0]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "chunk_size": self.chunk_size,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout,
        }

    def get_available_models(self) -> List[str]:
        """Get list of available embedding models from Llama Stack."""
        try:
            models = self.client.models.list()
            embedding_models = [
                model.identifier
                for model in models
                if getattr(model, "model_type", "") == "embedding"
            ]
            return embedding_models
        except Exception as e:
            logger.error(f"Error fetching available embedding models: {e}")
            return []

    def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """Get information about a specific embedding model."""
        model_to_check = model_id or self.model
        try:
            models = self.client.models.list()
            for model in models:
                if model.identifier == model_to_check:
                    return {
                        "identifier": model.identifier,
                        "provider_resource_id": getattr(
                            model, "provider_resource_id", None
                        ),
                        "provider_id": getattr(model, "provider_id", None),
                        "model_type": getattr(model, "model_type", None),
                        "metadata": getattr(model, "metadata", {}),
                        "embedding_dimension": getattr(model, "metadata", {}).get(
                            "embedding_dimension"
                        ),
                        "context_length": getattr(model, "metadata", {}).get(
                            "context_length"
                        ),
                    }
            return {"error": f"Model {model_to_check} not found"}
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return {"error": str(e)}

    def get_embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the current model."""
        model_info = self.get_model_info()
        return model_info.get("embedding_dimension")

    def similarity_search_by_vector(
        self, embedding: List[float], documents: List[str], k: int = 4
    ) -> List[tuple]:
        """
        Find the most similar documents to a given embedding vector.

        Args:
            embedding: Query embedding vector
            documents: List of documents to search through
            k: Number of most similar documents to return

        Returns:
            List of tuples (document, similarity_score)
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            # Get embeddings for all documents
            doc_embeddings = self.embed_documents(documents)

            # Calculate similarity scores
            similarities = cosine_similarity([embedding], doc_embeddings)[0]

            # Get top k documents
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                results.append((documents[idx], float(similarities[idx])))

            return results

        except ImportError:
            raise ImportError(
                "numpy and scikit-learn are required for similarity search. "
                "Install with: pip install numpy scikit-learn"
            )
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
