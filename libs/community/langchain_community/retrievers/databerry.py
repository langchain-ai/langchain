from typing import List, Optional

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DataberryRetriever(BaseRetriever):
    """`Databerry API` retriever."""

    datastore_url: str
    top_k: Optional[int]
    api_key: Optional[str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
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
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
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
