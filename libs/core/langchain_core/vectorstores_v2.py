####################
# Interfaces
####################


class Index(ABC):
    """Interface for a document index."""

    @abstractmethod
    def add(self, documents: Sequence[Document], **kwargs: Any) -> AddResponse:
        """Add documents to index."""

    @abstractmethod
    def delete(
        self, *, ids: Optional[Union[List[str], Tuple[str]]] = None, **kwargs: Any
    ) -> DeleteResponse:
        """Delete documents by id."""

    @abstractmethod
    def get(
        self, *, ids: Optional[Union[List[str], Tuple[str]]] = None, **kwargs: Any
    ) -> GetResponse:
        """Get documents by id."""

    # QUESTION: do we need Index.update or Index.upsert? should Index.add just do that?
    # QUESTION: should we support lazy versions of operations?


Q = TypeVar("Q")


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

    def as_runnable(
        self,
        *,
        output_format: Literal["string", "documents", "hits"] = "string",
        document_formatter: Optional[
            Callable[Sequence[Document], str]
        ] = _document_formatter,
    ) -> RunnableSerializable:
        if output_format == "string":
            return self | (lambda res: document_formatter(res.documents))
        elif output_format == "documents":
            return self | (lambda res: res.documents)
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
    succeeded: int
    failed: int


class DeleteResponse(TypedDict):
    succeeded: int
    failed: int


GetResponse = List[Document]


class RetrievalResponse(TypedDict, total=False):
    hits: List[Hit]
    metadata: dict

    @property
    def documents(self) -> List[Document]:
        return [h.document for h in hits]


class Hit(TypedDict):
    source: Document
    snippet: Optional[Tuple[float, float]]
    id: str
    score: float
    metric: str
    extra: dict
