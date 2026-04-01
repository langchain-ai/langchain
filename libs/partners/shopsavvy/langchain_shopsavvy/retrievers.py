"""Retriever using the ShopSavvy Data API."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator
from shopsavvy import ShopSavvyDataAPI  # type: ignore[import-untyped]

from langchain_shopsavvy._utilities import initialize_client


class ShopSavvyRetriever(BaseRetriever):
    """Retriever that searches ShopSavvy's product database.

    Setup:
        Install ``langchain-shopsavvy`` and set environment variable
        ``SHOPSAVVY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-shopsavvy
            export SHOPSAVVY_API_KEY="ss_live_your_api_key"

    Instantiation:
        .. code-block:: python

            from langchain_shopsavvy import ShopSavvyRetriever

            retriever = ShopSavvyRetriever(k=5)

    Usage:
        .. code-block:: python

            docs = retriever.invoke("iphone 15 pro")
            for doc in docs:
                print(doc.page_content)
                print(doc.metadata)
    """

    k: int = 5
    """Number of product results to return (1 to 100)."""

    client: ShopSavvyDataAPI = Field(default=None)  # type: ignore[assignment]
    shopsavvy_api_key: SecretStr = Field(default=SecretStr(""))

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the client."""
        return initialize_client(values)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Retrieve product documents matching the query.

        Args:
            query: The search query string.
            run_manager: The run manager for callbacks.

        Returns:
            List of Documents with product data as page_content and metadata.
        """
        result = self.client.search_products(query, limit=self.k)
        documents = []
        for product in result.data:
            page_content = json.dumps(
                {
                    "title": product.title,
                    "brand": product.brand,
                    "category": product.category,
                    "description": product.description,
                },
                indent=2,
            )
            metadata: dict[str, Any] = {
                "shopsavvy_id": product.shopsavvy,
                "barcode": product.barcode,
                "asin": product.amazon,
                "model": product.model,
                "brand": product.brand,
                "category": product.category,
            }
            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )
        return documents
