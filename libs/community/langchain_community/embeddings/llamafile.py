import logging
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LlamafileEmbeddings(BaseModel, Embeddings):
    """Llamafile lets you distribute and run large language models with a
    single file.

    To get started, see: https://github.com/Mozilla-Ocho/llamafile

    To use this class, you will need to first:

    1. Download a llamafile.
    2. Make the downloaded file executable: `chmod +x path/to/model.llamafile`
    3. Start the llamafile in server mode with embeddings enabled:

        `./path/to/model.llamafile --server --nobrowser --embedding`

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LlamafileEmbeddings
            embedder = LlamafileEmbeddings()
            doc_embeddings = embedder.embed_documents(
                [
                    "Alpha is the first letter of the Greek alphabet",
                    "Beta is the second letter of the Greek alphabet",
                ]
            )
            query_embedding = embedder.embed_query(
                "What is the second letter of the Greek alphabet"
            )

    """

    base_url: str = "http://localhost:8080"
    """Base url where the llamafile server is listening."""

    request_timeout: Optional[int] = None
    """Timeout for server requests"""

    def _embed(self, text: str) -> List[float]:
        try:
            response = requests.post(
                url=f"{self.base_url}/embedding",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "content": text,
                },
                timeout=self.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to Llamafile server. Please make sure "
                f"that a server is running at {self.base_url}."
            )

        # Raise exception if we got a bad (non-200) response status code
        response.raise_for_status()

        contents = response.json()
        if "embedding" not in contents:
            raise KeyError(
                "Unexpected output from /embedding endpoint, output dict "
                "missing 'embedding' key."
            )

        embedding = contents["embedding"]

        # Sanity check the embedding vector:
        # Prior to llamafile v0.6.2, if the server was not started with the
        # `--embedding` option, the embedding endpoint would always return a
        # 0-vector. See issue:
        # https://github.com/Mozilla-Ocho/llamafile/issues/243
        # So here we raise an exception if the vector sums to exactly 0.
        if sum(embedding) == 0.0:
            raise ValueError(
                "Embedding sums to 0, did you start the llamafile server with "
                "the `--embedding` option enabled?"
            )

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a llamafile server running at `self.base_url`.
        llamafile server should be started in a separate process before invoking
        this method.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        doc_embeddings = []
        for text in texts:
            doc_embeddings.append(self._embed(text))
        return doc_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a llamafile server running at `self.base_url`.
        llamafile server should be started in a separate process before invoking
        this method.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed(text)
