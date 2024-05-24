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
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.stores import BaseStore

####################
# INTERFACES
####################


class Index(BaseStore[str, Document], ABC):
    """Interface for a document index."""

    @abstractmethod
    def add(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Union[List[str], Tuple[str]]] = None,
        **kwargs: Any,
    ) -> AddResponse:
        """Add documents to index."""

    @abstractmethod
    def delete(
        self, *, ids: Optional[Union[List[str], Tuple[str]]] = None, **kwargs: Any
    ) -> DeleteResponse:
        """Delete documents."""

    @abstractmethod
    def get(
        self, *, ids: Optional[Union[List[str], Tuple[str]]] = None, **kwargs: Any
    ) -> GetResponse:
        """Get documents."""

    @abstractmethod
    def delete_by_ids(self, ids: Union[List[str], Tuple[str]]) -> DeleteResponse:
        """Delete documents by id."""

    @abstractmethod
    def get_by_ids(self, ids: Union[List[str], Tuple[str]]) -> GetResponse:
        """Get documents by id."""

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return cast(List[Optional[Document]], self.get_by_ids(keys))  # type: ignore[arg-type]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        ids, documents = zip(*key_value_pairs)
        self.add(documents, ids=ids)

    def mdelete(self, keys: Sequence[str]) -> None:
        self.delete_by_ids(keys)  # type: ignore[arg-type]

    # QUESTION: do we need Index.update or Index.upsert? should Index.add just do that?
    # QUESTION: should we support lazy versions of operations?


Q = TypeVar("Q")


class RetrievalResponse(TypedDict, total=False):
    hits: List[Hit]
    metadata: dict


class RetrieverV2(
    RunnableSerializable[Union[Q, Dict[str, Any]], RetrievalResponse], Generic[Q], ABC
):
    """Interface for a document retriever."""

    @abstractmethod
    def retrieve(self, query: Q, **kwargs: Any) -> RetrievalResponse:
        """Retrieve documents by query."""

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
        result = self.retrieve(**params)
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
            raise ValueError


class VectorStoreV2(Index, RetrieverV2[Union[str, Sequence[float]]]):
    @abstractmethod
    def retrieve(
        self,
        query: Union[str, Sequence[float]],
        *,
        method: str = "similarity",
        metric: str = "cosine_similarity",
        **kwargs: Any,
    ) -> RetrievalResponse:
        """Retrieve documents by query"""


####################
# TYPES
####################


class AddResponse(TypedDict):
    succeeded: List[str]
    failed: List[str]


class DeleteResponse(TypedDict):
    succeeded: List[str]
    failed: List[str]


GetResponse = List[Document]


class Hit(TypedDict, total=False):
    source: Optional[Document]
    snippet: Optional[str]
    id: str
    score: float
    metric: str
    extra: dict


####################
# HELPERS
####################


def _default_document_formatter(documents: Sequence[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def _retrieval_response_to_documents(response: RetrievalResponse) -> List[Document]:
    return [hit["source"] or Document(hit["snippet"]) for hit in response["hits"]]
