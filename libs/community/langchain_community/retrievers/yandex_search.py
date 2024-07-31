from typing import Any, Dict, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper


class YandexSearchAPIRetriever(BaseRetriever):
    """Yandex Search API retriever."""

    api_wrapper: YandexSearchAPIWrapper = Field(default_factory=YandexSearchAPIWrapper)  # type: ignore[arg-type]
    k: int = 10

    @staticmethod
    def _generate_documents(results: List[Dict[str, Any]]) -> List[Document]:
        docs = []

        for result in results:
            doc = Document(
                page_content=result.pop("content", ""),
                metadata=result,
            )
            docs.append(doc)

        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.api_wrapper.results(query)
        docs = self._generate_documents(results)
        return docs[: self.k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = await self.api_wrapper.results_async(query)
        docs = self._generate_documents(results)
        return docs[: self.k]
