"""Chat Model Components Derived from ChatModel/NVAIPlay"""

import asyncio
from typing import Any, Dict, List, Literal, Optional

from langchain.llms.nv_aiplay import ClientModel, NVCRModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema.embeddings import Embeddings

## NOTE: This file should not be ran in isolation as a single-file standalone.
## Please use llms.nv_aiplay instead.


class NVAIPlayEmbeddings(ClientModel, Embeddings):
    """NVIDIA's AI Playground NVOLVE Question-Answer Asymmetric Model."""

    client: NVCRModel = Field(NVCRModel)
    model_name: str = Field("nvolveqa")
    model: Optional[str] = Field(None)
    max_length: int = Field(2048, ge=1, le=2048)

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["client"] = values["client"](**values)
        model_name = values.get("model")
        model_name = model_name if model_name else values["model_name"]
        values["model_name"] = model_name
        values["model"] = model_name
        return values

    def _embed(self, text: str, model_type: Literal["passage", "query"]) -> List[float]:
        """Embed a single text entry to either passage or query type"""
        if len(text) > self.max_length:
            text = text[: self.max_length]
        output = self.client.get_req_generation(
            model_name=self.model_name,
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

    # async def aembed_query(self, text: str) -> List[float]:
    #     """Asynchronous Embed query text."""
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, self.embed_query, text
    #     )

    ## Implemented in Embeddings core class
    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Asynchronous Embed search docs."""
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, self.embed_documents, texts
    #     )

    async def aembed_batch_query(self, texts: List[str]) -> List[List[float]]:
        """Embed query text with Asynchronous Batching."""
        queries = [self.aembed_query(text) for text in texts]
        return await asyncio.gather(*queries)

    async def aembed_batch_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs with Asynchronous Batching."""
        passages = [self.aembed_documents([text]) for text in texts]
        outs = await asyncio.gather(*passages)
        return [out[0] for out in outs]
