from typing import List

import numpy as np

from langchain.embeddings.base import Embeddings


class FakeEmbeddings(Embeddings):
    size: int

    # Add float precision so we can specify 32 or 64 bit floats
    precision: str = "float64"

    def _get_embedding(self) -> List[float]:
        if self.precision == "float32":
            return list(np.random.normal(size=self.size).astype(np.float32))
        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding() for _ in texts]

    def _embed_query(self, text: str) -> List[float]:
        return self._get_embedding()
