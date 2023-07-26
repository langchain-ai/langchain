from typing import Any, Dict, List, Optional
from pydantic import BaseModel, root_validator
from langchain.embeddings.base import Embeddings

class AwaEmbeddings(BaseModel, Embeddings):
    client: Any  #: :meta private:
    model: str = "all-mpnet-base-v2"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that awadb library is installed."""

        try:
            import sys
            sys.path.insert(0, "/Users/taozhiwang/Desktop/Files/repo/awadb")
            from awadb import AwaEmbedding
        except ImportError as exc:
            raise ImportError(
                "Could not import awadb library. "
                "Please install it with `pip install awadb`"
            ) from exc
        values["client"] = AwaEmbedding()
        return values

    def set_model(model_name):
        model = model_name
        client = AwaEmbeddings(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.EmbeddingBatch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.Embedding(text)