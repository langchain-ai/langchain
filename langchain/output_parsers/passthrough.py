from __future__ import annotations

from typing import Generic, Optional, TypeVar, TypedDict

from langchain.schema import BaseOutputParser, OutputParserException

T = TypeVar("T")


class PassthroughOutput(TypedDict, Generic[T]):
    parsed: Optional[T]
    error: Optional[Exception]
    raw: str


class PassthroughOutputParser(BaseOutputParser[PassthroughOutput[T]]):
    """Wraps a parser and tries to fix parsing errors."""

    parser: BaseOutputParser[T]

    def parse(self, completion: str) -> PassthroughOutput[T]:
        try:
            parsed_completion = self.parser.parse(completion)
            return PassthroughOutput(
                parsed=parsed_completion, error=None, raw=completion
            )
        except OutputParserException as e:
            return PassthroughOutput(parsed=None, error=e, raw=completion)

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "passthrough"
