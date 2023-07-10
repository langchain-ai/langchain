from typing import Any, List, Optional

import aiohttp
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class ChaindeskRetriever(BaseRetriever):
    """Retriever that uses the Chaindesk API."""

    datastore_url: str
    top_k: Optional[int]
    api_key: Optional[str]

    def __init__(
        self,
        datastore_url: str,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        self.datastore_url = datastore_url
        self.api_key = api_key
        self.top_k = top_k

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        response = requests.post(
            self.datastore_url,
            json={
                "query": query,
                **({"topK": self.top_k} if self.top_k is not None else {}),
            },
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key is not None
                    else {}
                ),
            },
        )
        data = response.json()
        return [
            Document(
                page_content=r["text"],
                metadata={"source": r["source"], "score": r["score"]},
            )
            for r in data["results"]
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "POST",
                self.datastore_url,
                json={
                    "query": query,
                    **({"topK": self.top_k} if self.top_k is not None else {}),
                },
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key is not None
                        else {}
                    ),
                },
            ) as response:
                data = await response.json()
        return [
            Document(
                page_content=r["text"],
                metadata={"source": r["source"], "score": r["score"]},
            )
            for r in data["results"]
        ]
