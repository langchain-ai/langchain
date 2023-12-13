"""Embeddings Components Derived from ChatModel/NVAIPlay"""
from typing import Any, List, Literal, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field

import langchain_nvidia_aiplay._common as nvaiplay_common


class NVAIPlayEmbeddings(Embeddings):
    """NVIDIA's AI Playground NVOLVE Question-Answer Asymmetric Model."""

    client: nvaiplay_common.NVCRModel = Field(nvaiplay_common.NVCRModel)
    model: str = Field("nvolveqa")
    max_length: int = Field(2048, ge=1, le=2048)

    def __init__(self, *args: Sequence, **kwargs: Any):
        if "client" not in kwargs:
            kwargs["client"] = nvaiplay_common.NVCRModel(**kwargs)
        super().__init__(*args, **kwargs)

    def _embed(self, text: str, model_type: Literal["passage", "query"]) -> List[float]:
        """Embed a single text entry to either passage or query type"""
        if len(text) > self.max_length:
            text = text[: self.max_length]
        output = self.client.get_req_generation(
            model_name=self.model,
            payload={
                "input": text,
                "model": model_type,
                "encoding_format": "float",
            },
        )
        return output.get("embedding", [])

    def embed_query(self, text: str) -> List[float]:
        """Input pathway for query embeddings."""
        return self._embed(text, model_type="query")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Input pathway for document embeddings."""
        return [self._embed(text, model_type="passage") for text in texts]
