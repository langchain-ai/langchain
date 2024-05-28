from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    run_in_executor,
)

Q = TypeVar("Q")


class RetrievalResponse(TypedDict, total=False):
    hits: List[Hit]
    metadata: dict


class RetrieverV2(
    RunnableSerializable[Union[Q, Dict[str, Any]], RetrievalResponse], Generic[Q], ABC
):
    """Interface for a document retriever."""

    @abstractmethod
    def _retrieve(self, query: Q, **kwargs: Any) -> RetrievalResponse:
        """Retrieve documents by query."""

    async def _aretrieve(self, query: Q, **kwargs: Any) -> RetrievalResponse:
        """Retrieve documents by query."""
        return await run_in_executor(
            None,
            self._retrieve,
            query,
            **kwargs,
        )

    def invoke(
        self,
        input: Union[Q, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> RetrievalResponse:
        # TODO: callback manager stuff
        if isinstance(input, Dict):
            params = {**input, **kwargs}
        else:
            params = {"query": input, **kwargs}
        result = self._retrieve(**params)
        # TODO: callback manager stuff
        return result

    async def ainvoke(
        self,
        input: Union[Q, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> RetrievalResponse:
        # TODO: callback manager stuff
        if isinstance(input, Dict):
            params = {**input, **kwargs}
        else:
            params = {"query": input, **kwargs}
        result = await self._aretrieve(**params)
        # TODO: callback manager stuff
        return result

    # QUESTION: Anything to do with streaming?

    def with_output_format(
        self,
        *,
        output_format: Literal["string", "documents", "hits"] = "string",
        document_formatter: Optional[Callable[[Sequence[Document]], str]] = None,
    ) -> RunnableSerializable:
        if output_format == "string":
            document_formatter = document_formatter or _default_document_formatter
            return self | _retrieval_response_to_documents | document_formatter
        elif output_format == "documents":
            return self | _retrieval_response_to_documents
        elif output_format == "hits":
            return self
        else:
            raise ValueError(
                f"Unsupported {output_format=}. Expected one of ['string', "
                "'documents', 'hits']."
            )


class Hit(TypedDict, total=False):
    source: Optional[Document]
    snippet: Optional[str]
    id: str
    score: float
    metric: str
    extra: dict


def _default_document_formatter(documents: Sequence[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def _retrieval_response_to_documents(response: RetrievalResponse) -> List[Document]:
    return [
        hit["source"] or Document(cast(str, hit["snippet"])) for hit in response["hits"]
    ]
