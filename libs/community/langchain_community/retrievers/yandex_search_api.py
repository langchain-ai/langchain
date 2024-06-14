from typing import Any, Dict, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities import YandexSearchAPIClient


class YandexSearchAPIRetriever(BaseRetriever):
    """Yandex Search API retriever."""

    client: YandexSearchAPIClient
    k: int = 10

    @staticmethod
    def _generate_documents(response: List[Dict[str, Any]]) -> List[Document]:
        docs = []

        for result in response:
            passages = result["passages"]

            if passages:
                if isinstance(passages, str):
                    text = passages
                elif isinstance(passages, list):
                    text = "\n".join(passages)
                text_type = "passages"
            else:
                text = result["headline"]
                text_type = "headline"

            doc = Document(
                page_content=text,
                metadata={
                    "modified_at": result["modified_at"].strftime("%Y-%m-%d %H:%M:%S"),
                    "page_content_type": text_type,
                    "title": result["title"],
                    "url": result["url"],
                    "saved_copy_url_html": f"{result['saved_copy_url']}&mode=html",
                    "saved_copy_url_text": f"{result['saved_copy_url']}&mode=text",
                },
            )
            docs.append(doc)

        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        response = self.client.search(query=query)
        docs = self._generate_documents(response)
        return docs[: self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        response = await self.client.asearch(query=query)
        docs = self._generate_documents(response)
        return docs[: self.k]
