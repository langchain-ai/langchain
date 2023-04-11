from typing import List, Optional

import aiohttp
import requests

from langchain.schema import BaseRetriever, Document


class DataberryRetriever(BaseRetriever):
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

    def get_relevant_documents(self, query: str) -> List[Document]:
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

    async def aget_relevant_documents(self, query: str) -> List[Document]:
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
