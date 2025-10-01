"""Baseten embeddings wrapper."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self


class BasetenEmbeddings(BaseModel, Embeddings):
    """Baseten embedding model integration.

    Baseten embedding model integration using Performance Client for optimized
    embedding generation with automatic batching, concurrent requests, and
    intelligent request sizing.

    Setup:
        Install ``langchain-baseten`` and set environment variable ``BASETEN_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-baseten
            export BASETEN_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Baseten model to use.
        model_url: str
            The specific model URL for your deployed embedding model.
            Compatible with /sync, /sync/v1, or /predict endpoints with built-in error correction.

    Key init args — client params:
        baseten_api_key: SecretStr
            Baseten API key. If not passed in will be read from env var ``BASETEN_API_KEY``.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_baseten import BasetenEmbeddings

            # All of these URL formats work with automatic error correction:

            # Option 1: /sync endpoint (recommended)
 again            embeddings = BasetenEmbeddings(
                model="your-embedding-model",
                model_url="https://model-<id>.api.baseten.co/environments/production/sync",
            )

            # Option 2: /sync/v1 endpoint (automatically normalized)
            embeddings = BasetenEmbeddings(
                model="your-embedding-model",
                model_url="https://model-<id>.api.baseten.co/environments/production/sync/v1",
            )

            # Option 3: /predict endpoint (automatically converted to /sync)
            embeddings = BasetenEmbeddings(
                model="your-embedding-model",
                model_url="https://model-<id>.api.baseten.co/environments/production/predict",
            )

    Embed multiple texts:
        .. code-block:: python

            texts = ["hello world", "goodbye world", "machine learning"]
            vectors = embeddings.embed_documents(texts)
            print(f"Generated {len(vectors)} embeddings")
            print(f"Each embedding has {len(vectors[0])} dimensions")

        .. code-block:: python

            Generated 3 embeddings
            Each embedding has 768 dimensions

    Embed single text:
        .. code-block:: python

            query = "What is artificial intelligence?"
            vector = embeddings.embed_query(query)
            print(f"Query embedding has {len(vector)} dimensions")
            print(f"First few values: {vector[:3]}")

        .. code-block:: python

            Query embedding has 768 dimensions
            First few values: [-0.021892, -0.015861, 0.012778]

    Async usage:
        .. code-block:: python

            # Async embedding for better performance in async applications
            vectors = await embeddings.aembed_documents(["hello", "goodbye"])
            vector = await embeddings.aembed_query("hello")

    Batch processing with Performance Client:
        .. code-block:: python

            # Performance Client automatically handles large batches efficiently
            large_text_list = ["text " + str(i) for i in range(1000)]
            vectors = embeddings.embed_documents(large_text_list)
            print(f"Processed {len(vectors)} embeddings with automatic batching")

        .. code-block:: python

            Processed 1000 embeddings with automatic batching

    Integration with vector stores:
        .. code-block:: python

            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            # Create documents
            docs = [
                Document(page_content="Cats are independent pets"),
                Document(page_content="Dogs are loyal companions"),
                Document(page_content="Birds can fly and sing"),
            ]

            # Create vector store with Baseten embeddings
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Search for similar documents
            results = vectorstore.similarity_search("pets and animals", k=2)
            for doc in results:
                print(doc.page_content)

        .. code-block:: python

            Cats are independent pets
            Dogs are loyal companions

    Performance features:
        The Performance Client provides several optimizations and built-in error correction:

        - **Automatic batching**: Processes texts in optimal batch sizes (32)
        - **Concurrent requests**: Up to 128 concurrent requests for large datasets
        - **Smart request sizing**: Limits requests to 8000 characters for optimal performance
        - **Error handling**: Robust retry logic and error management
        - **URL normalization**: Handles various endpoint URL formats automatically
        - **Built-in error correction**: Automatically converts between /sync, /sync/v1, and /predict endpoints
        - **Endpoint compatibility**: Works with any Baseten model endpoint format

    Note:
        Unlike chat models which use the general Model APIs, Baseten embeddings
        require a specific model URL that points to your deployed embedding model.
        You can find this URL in your Baseten dashboard.

        **Supported URL formats** (with automatic error correction):

        - ``https://model-<id>.api.baseten.co/environments/production/sync``
        - ``https://model-<id>.api.baseten.co/environments/production/sync/v1``
        - ``https://model-<id>.api.baseten.co/environments/production/predict``

        The integration automatically normalizes URLs and handles endpoint differences,
        so you can use any of these formats and it will work correctly.
    """

    client: Any = Field(default=None, exclude=True)  # :meta private:
    baseten_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "BASETEN_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `baseten_api_key=...` or "
                "set the environment variable `BASETEN_API_KEY`."
            ),
        ),
    )
    """Baseten API key. Automatically read from env variable ``BASETEN_API_KEY`` if not provided."""

    model: str = Field(default="embeddings")
    """Model name to use for embeddings."""

    model_url: str = Field(...)
    """The specific model URL for your deployed embedding model.

    This should be the /sync or /sync/v1 endpoint URL from your Baseten dashboard.
    Example: https://model-<id>.api.baseten.co/environments/production/predict
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate environment variables and setup client."""
        try:
            from baseten_performance_client import PerformanceClient
        except ImportError as e:
            msg = (
                "Could not import baseten_performance_client python package. "
                "Please install it with `pip install baseten_performance_client`."
            )
            raise ImportError(msg) from e

        # Ensure the model_url is provided
        if not self.model_url:
            msg = (
                "model_url is required for Baseten embeddings. "
                "Please provide the endpoint URL from your Baseten dashboard. "
                "Supports /sync, /sync/v1, or /predict endpoints with automatic error correction."
            )
            raise ValueError(msg)

        # Normalize the URL for Performance Client
        # Strip /v1 from URL if present to just end in /sync, as Performance Client handles this format
        # Strip /predict from URL if present and replace with /sync, as Performance Client handles this format
        base_url = self.model_url
        if base_url.endswith("/predict"):
            base_url = base_url.replace("/predict", "/sync")
        if base_url.endswith("/sync/v1"):
            base_url = base_url.replace("/sync/v1", "/sync")
        elif base_url.endswith("/v1"):
            base_url = base_url[:-3]

        # Ensure URL ends with /sync
        if not base_url.endswith("/sync"):
            if base_url.endswith("/"):
                base_url = f"{base_url}sync"
            else:
                base_url = f"{base_url}/sync"

        # Create Performance Client
        self.client = PerformanceClient(
            base_url=base_url,
            api_key=self.baseten_api_key.get_secret_value()
        )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        try:
            # Use Performance Client's embed method with optimized batching
            response = self.client.embed(
                input=texts,
                model=self.model,
                batch_size=32,  # Optimize batch size for performance
                max_concurrent_requests=128,
                max_chars_per_request=8000,
            )

            # Performance Client returns response.data with embeddings
            return [item.embedding for item in response.data]

        except Exception as e:
            msg = f"Error calling Baseten embeddings API: {e}"
            raise RuntimeError(msg) from e

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed search docs.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        try:
            # Use Performance Client's async embed method
            response = await self.client.async_embed(
                input=texts,
                model=self.model,
                batch_size=32,
                max_concurrent_requests=128,
                max_chars_per_request=8000,
            )

            # Performance Client returns response.data with embeddings
            return [item.embedding for item in response.data]

        except Exception as e:
            msg = f"Error calling Baseten embeddings API: {e}"
            raise RuntimeError(msg) from e

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding for the text.
        """
        result = await self.aembed_documents([text])
        return result[0]
