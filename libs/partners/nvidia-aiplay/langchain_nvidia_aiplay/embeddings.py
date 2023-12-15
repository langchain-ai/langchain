"""Embeddings Components Derived from ChatModel/NVAIPlay"""
from typing import Any, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

import langchain_nvidia_aiplay._common as nvaiplay_common


class NVAIPlayEmbeddings(BaseModel, Embeddings):
    """NVIDIA's AI Playground NVOLVE Question-Answer Asymmetric Model."""

    client: nvaiplay_common.NVCRModel = Field(nvaiplay_common.NVCRModel)
    model: str = Field(
        ..., description="The embedding model to use. Example: nvolveqa_40k"
    )
    max_length: int = Field(2048, ge=1, le=2048)
    max_batch_size: int = Field(default=50)
    model_type: Optional[Literal["passage", "query"]] = Field(
        "passage", description="The type of text to be embedded."
    )

    @root_validator(pre=True)
    def _validate_client(cls, values: Any) -> Any:
        if "client" not in values:
            values["client"] = nvaiplay_common.NVCRModel()
        return values

    @property
    def available_models(self) -> dict:
        """Map the available models that can be invoked."""
        return self.client.available_models

    def _embed(
        self, texts: List[str], model_type: Literal["passage", "query"]
    ) -> List[List[float]]:
        """Embed a single text entry to either passage or query type"""
        response = self.client.get_req(
            model_name=self.model,
            payload={
                "input": texts,
                "model": model_type,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        result = response.json()
        data = result["data"]
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of embeddings. Got: {data}")
        embedding_list = [(res["embedding"], res["index"]) for res in data]
        return [x[0] for x in sorted(embedding_list, key=lambda x: x[1])]

    def embed_query(self, text: str) -> List[float]:
        """Input pathway for query embeddings."""
        return self._embed([text], model_type=self.model_type or "query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Input pathway for document embeddings."""
        # From https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-40k/documentation
        # The input must not exceed the 2048 max input characters and inputs above 512
        # model tokens will be truncated. The input array must not exceed 50 input
        #  strings.
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            truncated = [
                text[: self.max_length] if len(text) > self.max_length else text
                for text in batch
            ]
            all_embeddings.extend(
                self._embed(truncated, model_type=self.model_type or "passage")
            )
        return all_embeddings
