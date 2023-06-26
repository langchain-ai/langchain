from typing import Any, List, Optional

from langchain.schema import BaseRetriever, Document


class MetalRetriever(BaseRetriever):
    """Retriever that uses the Metal API."""

    def __init__(self, client: Any, params: Optional[dict] = None):
        from metal_sdk.metal import Metal

        if not isinstance(client, Metal):
            raise ValueError(
                "Got unexpected client, should be of type metal_sdk.metal.Metal. "
                f"Instead, got {type(client)}"
            )
        self.client: Metal = client
        self.params = params or {}

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.client.search({"text": query}, **self.params)
        final_results = []
        for r in results["data"]:
            metadata = {k: v for k, v in r.items() if k != "text"}
            final_results.append(Document(page_content=r["text"], metadata=metadata))
        return final_results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
