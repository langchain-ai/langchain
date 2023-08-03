from __future__ import annotations

from typing import List, Optional

import aiohttp
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class ChatGPTPluginRetriever(BaseRetriever):
    """Retrieves documents from a ChatGPT plugin."""

    url: str
    """URL of the ChatGPT plugin."""
    bearer_token: str
    """Bearer token for the ChatGPT plugin."""
    top_k: int = 3
    """Number of documents to return."""
    filter: Optional[dict] = None
    """Filter to apply to the results."""
    aiosession: Optional[aiohttp.ClientSession] = None
    """Aiohttp session to use for requests."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        """Allow arbitrary types."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        url, json, headers = self._create_request(query)
        response = requests.post(url, json=json, headers=headers)
        results = response.json()["results"][0]["results"]
        docs = []
        for d in results:
            content = d.pop("text")
            metadata = d.pop("metadata", d)
            if metadata.get("source_id"):
                metadata["source"] = metadata.pop("source_id")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        url, json, headers = self._create_request(query)

        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=json) as response:
                    res = await response.json()
        else:
            async with self.aiosession.post(
                url, headers=headers, json=json
            ) as response:
                res = await response.json()

        results = res["results"][0]["results"]
        docs = []
        for d in results:
            content = d.pop("text")
            metadata = d.pop("metadata", d)
            if metadata.get("source_id"):
                metadata["source"] = metadata.pop("source_id")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _create_request(self, query: str) -> tuple[str, dict, dict]:
        url = f"{self.url}/query"
        json = {
            "queries": [
                {
                    "query": query,
                    "filter": self.filter,
                    "top_k": self.top_k,
                }
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
        }
        return url, json, headers
