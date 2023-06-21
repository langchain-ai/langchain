"""Multi strategy output parser."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterator, Sequence, TypeVar, Union, Optional

from langchain.schema import (
    BaseOutputParser,
    OutputParserException,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")

TPredicate = Callable[[str], bool]
TParser = Callable[[str], S]


class ParseStrategy(Generic[S]):
    """A strategy is a pair of (parser, predicate).

    This class behave like a tuple for easy definition of multiple strategies.
    """

    def __init__(
        self, parser: TParser[S], predicate: TPredicate, name: Optional[str] = None
    ):
        assert callable(parser), "first argument <parser> must be callable"
        self.parser = parser
        assert callable(predicate), "second argument <predicate> must be callable"
        self.predicate = predicate
        self.name = name

    def __repr__(self) -> str:
        if self.name is None:
            return f"ParseStrategy(parser={self.parser}," "predicate={self.predicate})"
        return (
            f"ParseStrategy[{self.name}](parser={self.parser},"
            "predicate={self.predicate})"
        )

    def __getitem__(self, index: int) -> Union[TParser[S], TPredicate]:
        """Behaves like a tuple."""
        if index == 0:
            return self.parser
        elif index == 1:
            return self.predicate
        else:
            raise IndexError("tuple index out of range")

    def __iter__(self) -> Iterator[Any]:
        """Implement tuple unpacking."""
        yield self.parser
        yield self.predicate


class MultiStrategyParser(BaseOutputParser[T], ABC, Generic[T, S]):
    """Try multiple strategies to parse the output.

    A strategy is a tuple of (parser, predicate). The parser takes the some
    text as input and returns some type S. The parser is only called if the
    predicate returns True.

    When the `parse` method is called, all registered strategies are tried
    in order and the first one that succeeds returns its result.

    The returned value of type `S` is then passed to the final_parse method to
    produce the final result compatible with the inhertited output parser
    interface.

    Appending a strategy to the end makes it a fallback strategy.
    """

    class Config:
        arbitrary_types_allowed = True

    strategies: Sequence[ParseStrategy[S]]
    """List of strategies to try. The first one that succeeds is returned."""

    def add_strategy(self, *strategy: ParseStrategy[S]) -> None:
        """Register a new strategy.

        A strategy is a callbale that takes in text as `str` and returns
        some type `S`.
        """
        self.strategies = [*self.strategies, *strategy]

    @abstractmethod
    def final_parse(self, text: str, parsed: S) -> T:
        """Parse the output of a strategy."""

    def parse(self, text: str) -> T:
        """Try the registered strategies in order.

        Returns the output of the first succeeding strategy."""

        if len(self.strategies) == 0:
            raise OutputParserException("No strategy available")
        for strategy, predicate in self.strategies:
            log.debug(f"trying strategy {strategy}")
            if not predicate(text):
                log.debug(f"Skipping strategy {strategy}")
            if predicate(text):
                try:
                    parsed = strategy(text)
                    result = self.final_parse(text, parsed)
                    log.debug(f"Strategy {strategy} succeeded")
                    return result
                except Exception:
                    continue

        raise OutputParserException(f"Could not parse output: {text}")

    @property
    def _type(self) -> str:
        return "multi_strategy"
