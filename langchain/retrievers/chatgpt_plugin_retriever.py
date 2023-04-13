from typing import List, Optional

import aiohttp
import requests
from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document


class ChatGPTPluginRetriever(BaseRetriever, BaseModel):
    url: str
    bearer_token: str
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_relevant_documents(
        self,
        query: str,
        filter: Optional[BaseModel] = None,
        top_k: Optional[int] = 3,
    ) -> List[Document]:
        response = requests.post(
            f"{self.url}/query",
            json={"queries": [{"query": query, "filter": filter, "top_k": top_k}]},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            },
        )
        results = response.json()["results"][0]["results"]
        docs = []
        for d in results:
            content = d.pop("text")
            docs.append(Document(page_content=content, metadata=d))
        return docs

    async def aget_relevant_documents(
        self,
        query: str,
        filter: Optional[BaseModel] = None,
        top_k: Optional[int] = 3,
    ) -> List[Document]:
        url = f"{self.url}/query"
        json = ({"queries": [{"query": query, "filter": filter, "top_k": top_k}]},)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
        }

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
            docs.append(Document(page_content=content, metadata=d))
        return docs
