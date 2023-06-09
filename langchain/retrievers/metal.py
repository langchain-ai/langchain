from typing import Any, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class MetalRetriever(BaseRetriever):
    def __init__(self, client: Any, params: Optional[dict] = None):
        from metal_sdk.metal import Metal

        if not isinstance(client, Metal):
            raise ValueError(
                "Got unexpected client, should be of type metal_sdk.metal.Metal. "
                f"Instead, got {type(client)}"
            )
        self.client: Metal = client
        self.params = params or {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.client.search({"text": query}, **self.params)
        final_results = []
        for r in results["data"]:
            metadata = {k: v for k, v in r.items() if k != "text"}
            final_results.append(Document(page_content=r["text"], metadata=metadata))
        return final_results

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError
