from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseRetriever, Document


class MetalRetriever(BaseRetriever):
    """`Metal API` retriever."""

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
