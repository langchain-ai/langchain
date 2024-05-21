from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE_CHARS = 1000000
MAX_BATCH_SIZE_PARTS = 90


class GigaChatEmbeddings(BaseModel, Embeddings):
    """GigaChat Embeddings models.

    Example:
        .. code-block:: python
            from langchain_community.embeddings.gigachat import GigaChatEmbeddings

            embeddings = GigaChatEmbeddings(
                credentials=..., scope=..., verify_ssl_certs=False
            )
    """

    base_url: Optional[str] = None
    """ Base API URL """
    auth_url: Optional[str] = None
    """ Auth URL """
    credentials: Optional[str] = None
    """ Auth Token """
    scope: Optional[str] = None
    """ Permission scope for access token """

    access_token: Optional[str] = None
    """ Access token for GigaChat """

    model: Optional[str] = None
    """Model name to use."""
    user: Optional[str] = None
    """ Username for authenticate """
    password: Optional[str] = None
    """ Password for authenticate """

    timeout: Optional[float] = 600
    """ Timeout for request. By default it works for long requests. """
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None
    # Support for connection to GigaChat through SSL certificates

    @cached_property
    def _client(self) -> Any:
        """Returns GigaChat API client"""
        import gigachat

        return gigachat.GigaChat(
            base_url=self.base_url,
            auth_url=self.auth_url,
            credentials=self.credentials,
            scope=self.scope,
            access_token=self.access_token,
            model=self.model,
            user=self.user,
            password=self.password,
            timeout=self.timeout,
            verify_ssl_certs=self.verify_ssl_certs,
            ca_bundle_file=self.ca_bundle_file,
            cert_file=self.cert_file,
            key_file=self.key_file,
            key_file_password=self.key_file_password,
        )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate authenticate data in environment and python package is installed."""
        try:
            import gigachat  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import gigachat python package. "
                "Please install it with `pip install gigachat`."
            )
        fields = set(cls.__fields__.keys())
        diff = set(values.keys()) - fields
        if diff:
            logger.warning(f"Extra fields {diff} in GigaChat class")
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        result: List[List[float]] = []
        size = 0
        local_texts = []
        embed_kwargs = {}
        if self.model is not None:
            embed_kwargs["model"] = self.model
        for text in texts:
            local_texts.append(text)
            size += len(text)
            if size > MAX_BATCH_SIZE_CHARS or len(local_texts) > MAX_BATCH_SIZE_PARTS:
                for embedding in self._client.embeddings(
                    texts=local_texts, **embed_kwargs
                ).data:
                    result.append(embedding.embedding)
                size = 0
                local_texts = []
        # Call for last iteration
        if local_texts:
            for embedding in self._client.embeddings(
                texts=local_texts, **embed_kwargs
            ).data:
                result.append(embedding.embedding)

        return result

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        result: List[List[float]] = []
        size = 0
        local_texts = []
        embed_kwargs = {}
        if self.model is not None:
            embed_kwargs["model"] = self.model
        for text in texts:
            local_texts.append(text)
            size += len(text)
            if size > MAX_BATCH_SIZE_CHARS or len(local_texts) > MAX_BATCH_SIZE_PARTS:
                embeddings = await self._client.aembeddings(
                    texts=local_texts, **embed_kwargs
                )
                for embedding in embeddings.data:
                    result.append(embedding.embedding)
                size = 0
                local_texts = []
        # Call for last iteration
        if local_texts:
            embeddings = await self._client.aembeddings(
                texts=local_texts, **embed_kwargs
            )
            for embedding in embeddings.data:
                result.append(embedding.embedding)

        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a GigaChat embeddings models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents(texts=[text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query using a GigaChat embeddings models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        docs = await self.aembed_documents(texts=[text])
        return docs[0]
