from typing import List

import numpy as np
from pydantic import BaseModel

from langchain.embeddings.base import Embeddings
from langchain.schema import EmbeddingResult


class FakeEmbeddings(Embeddings):
    size: int

    def _get_embedding(self) -> List[float]:
        return list(np.random.normal(size=self.size))

    def _embed_documents(self, texts: List[str]) -> EmbeddingResult:
        embeddings = [self._get_embedding() for _ in texts]
        result = EmbeddingResult(
            embeddings=embeddings, llm_output={"token_usage": 100 * len(texts)}
        )
        return result

    def _embed_query(self, text: str) -> EmbeddingResult:
        embedding = self._get_embedding()
        result = EmbeddingResult(
            embeddings=[embedding], llm_output={"token_usage": 100}
        )
        return result
