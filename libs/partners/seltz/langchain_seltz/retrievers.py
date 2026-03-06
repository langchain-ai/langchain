"""Retriever using Seltz Search API."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator
from seltz import Includes, Seltz  # type: ignore[import-untyped]

from langchain_seltz._utilities import initialize_client


class SeltzSearchRetriever(BaseRetriever):
    """Seltz Search retriever.

    Setup:
        Install `langchain-seltz` and set environment variable `SELTZ_API_KEY`.

        ```bash
        pip install -U langchain-seltz
        export SELTZ_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_seltz import SeltzSearchRetriever

        retriever = SeltzSearchRetriever(k=5)
        ```

    Usage:
        ```python
        docs = retriever.invoke("what is the weather in SF")
        for doc in docs:
            print(doc.page_content)
            print(doc.metadata["url"])
        ```
    """

    k: int = 10
    """The number of documents to return."""

    client: Seltz = Field(default=None)  # type: ignore[assignment]
    seltz_api_key: SecretStr = Field(default=SecretStr(""))
    seltz_endpoint: str | None = None
    seltz_insecure: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment."""
        return initialize_client(values)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Get documents relevant to the query.

        Args:
            query: The search query.
            run_manager: The run manager for callbacks.

        Returns:
            A list of Documents with page_content and metadata.
        """
        response = self.client.search(
            query=query, includes=Includes(max_documents=self.k)
        )

        return [
            Document(
                page_content=doc.content,
                metadata={"url": doc.url},
            )
            for doc in response.documents
        ]
