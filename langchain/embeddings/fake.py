from typing import List, Sequence

import numpy as np
from pydantic import BaseModel

from langchain.callbacks.manager import (
    CallbackManagerForEmbeddingsRun,
)
from langchain.embeddings.base import Embeddings


class FakeEmbeddings(Embeddings, BaseModel):
    size: int

    def _get_embedding(self) -> List[float]:
        return list(np.random.normal(size=self.size))

    def _embed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[CallbackManagerForEmbeddingsRun],
    ) -> List[List[float]]:
        return [self._get_embedding() for _ in texts]

    def _embed_query(
        self,
        text: str,
        *,
        run_manager: CallbackManagerForEmbeddingsRun,
    ) -> List[float]:
        """Embed query text."""
        return self._get_embedding()
