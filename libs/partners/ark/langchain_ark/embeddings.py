import os
import logging
import warnings

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.embeddings import Embeddings


class ArkEmbeddings(BaseModel, Embeddings):
    """ARK embedding models.

    Example:

    """
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    query_encode_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            from volcenginesdkarkruntime import Ark  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import volcenginesdkarkruntime[ark] python package. "
                "Please install it with `pip install volcenginesdkarkruntime[ark]`."
            ) from exc

        self._client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        self._model = os.environ.get("ARK_EMBEDDING_MODEL")

    def _embed(self, texts: list[str], encode_kwargs: Dict[str, Any]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        resp = self._client.embeddings.create(
            model=self._model, input=texts, **encode_kwargs
        )

        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """

        :param texts: The list of texts to embed.
        :return: List of embeddings, one for each text.
        """

        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> List[float]:
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed([text], embed_kwargs)[0]
