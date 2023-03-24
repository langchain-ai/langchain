from typing import List

import requests
from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document


class ChatGPTPluginRetriever(BaseRetriever, BaseModel):
    url: str
    bearer_token: str

    def get_relevant_documents(self, query: str) -> List[Document]:
        response = requests.post(
            f"{self.url}/query",
            json={"queries": [{"query": query}]},
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
