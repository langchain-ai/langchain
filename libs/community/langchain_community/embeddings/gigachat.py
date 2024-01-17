"GigaChat embedddings access tool"
from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)


class GigaChatEmbeddings(BaseModel, Embeddings):
    """GigaChat Embeddings models.

    Example:
        .. code-block:: python
            from langchain_community.embeddings.gigachat import GigaChatEmbeddings

            embeddings = GigaChatEmbeddings(credentials=..., verify_ssl_certs=False)
    """

    one_by_one_mode: bool = True
    """ Send texts one-by-one to server (to increase token limit)"""

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

    timeout: Optional[float] = None
    """ Timeout for request """
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    _debug_delay: float = 0
    """ Debug timeout for limit rps to server"""

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
        fields.add("profanity_check")
        diff = set(values.keys()) - fields
        if diff:
            logger.warning(f"Extra fields {diff} in GigaChat class")
        if "profanity" in fields and values.get("profanity") is False:
            logger.warning(
                "Profanity field is deprecated. Use 'profanity_check' instead."
            )
            if values.get("profanity_check") is None:
                values["profanity_check"] = values.get("profanity")
        return values

    def embed_documents(
        self, texts: List[str], model="Embeddings"
    ) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.one_by_one_mode:
            result: List[List[float]] = []
            if self._debug_delay == 0:
                for text in texts:
                    for embedding in self._client.embeddings(
                        texts=[text], model=model
                    ).data:
                        result.append(embedding.embedding)
            else:
                for text in texts:
                    time.sleep(self._debug_delay)
                    for embedding in self._client.embeddings(
                        texts=[text], model=model
                    ).data:
                        result.append(embedding.embedding)
            return result
        else:
            return [
                embedding.embedding
                for embedding in self._client.embeddings(texts=texts, model=model).data
            ]

        # return _embed_with_retry(self, texts=texts)

    def embed_query(self, text: str, model="Embeddings") -> List[float]:
        """Embed a query using a GigaChat embeddings models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents(texts=[text], model=model)[0]
