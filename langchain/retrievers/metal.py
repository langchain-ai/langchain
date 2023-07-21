from typing import Any, List, Optional

from pydantic import root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class MetalRetriever(BaseRetriever):
    """Retriever that uses the Metal API."""

    client: Any
    """The Metal client to use."""
    params: Optional[dict] = None
    """The parameters to pass to the Metal client."""

    @root_validator(pre=True)
    def validate_client(cls, values: dict) -> dict:
        """Validate that the client is of the correct type."""
        from metal_sdk.metal import Metal

        if "client" in values:
            client = values["client"]
            if not isinstance(client, Metal):
                raise ValueError(
                    "Got unexpected client, should be of type metal_sdk.metal.Metal. "
                    f"Instead, got {type(client)}"
                )

        values["params"] = values.get("params", {})

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.client.search({"text": query}, **self.params)
        final_results = []
        for r in results["data"]:
            metadata = {k: v for k, v in r.items() if k != "text"}
            final_results.append(Document(page_content=r["text"], metadata=metadata))
        return final_results

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
