from typing import Callable, Iterable, Set, TypeVar

from langchain_core._api import beta

from langchain_community.graph_vectorstores import Link
from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)

InputT = TypeVar("InputT")
UnderlyingInputT = TypeVar("UnderlyingInputT")


@beta()
class LinkExtractorAdapter(LinkExtractor[InputT]):
    def __init__(
        self,
        underlying: LinkExtractor[UnderlyingInputT],
        transform: Callable[[InputT], UnderlyingInputT],
    ) -> None:
        self._underlying = underlying
        self._transform = transform

    def extract_one(self, input: InputT) -> Set[Link]:  # noqa: A002
        return self._underlying.extract_one(self._transform(input))

    def extract_many(self, inputs: Iterable[InputT]) -> Iterable[Set[Link]]:
        underlying_inputs = map(self._transform, inputs)
        return self._underlying.extract_many(underlying_inputs)
