from typing import List, Optional

import aiohttp
import requests
from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document


class RemoteLangChainRetriever(BaseRetriever, BaseModel):
    url: str
    headers: Optional[dict] = None
    input_key: str = "message"
    response_key: str = "response"
    page_content_key: str = "page_content"
    metadata_key: str = "metadata"

    def get_relevant_documents(self, query: str) -> List[Document]:
        response = requests.post(
            self.url, json={self.input_key: query}, headers=self.headers
        )
        result = response.json()
        return [
            Document(
                page_content=r[self.page_content_key], metadata=r[self.metadata_key]
            )
            for r in result[self.response_key]
        ]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "POST", self.url, headers=self.headers, json={self.input_key: query}
            ) as response:
                result = await response.json()
        return [
            Document(
                page_content=r[self.page_content_key], metadata=r[self.metadata_key]
            )
            for r in result[self.response_key]
        ]
