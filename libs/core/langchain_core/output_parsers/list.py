from __future__ import annotations

import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

T = TypeVar("T")


def droplastn(iter: Iterator[T], n: int) -> Iterator[T]:
    """Drop the last n elements of an iterator."""
    buffer: Deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()


class ListOutputParser(BaseTransformOutputParser[List[str]]):
    """Parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call."""
        raise NotImplementedError

    def _transform(
        self, input: Iterator[Union[str, BaseMessage]]
    ) -> Iterator[List[str]]:
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                # extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            # add current chunk to buffer
            buffer += chunk
            # parse buffer into a list of parts
            try:
                done_idx = 0
                # yield only complete parts
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # yield only complete parts
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # yield the last part
        for part in self.parse(buffer):
            yield [part]

    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[List[str]]:
        buffer = ""
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                # extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            # add current chunk to buffer
            buffer += chunk
            # parse buffer into a list of parts
            try:
                done_idx = 0
                # yield only complete parts
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # yield only complete parts
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # yield the last part
        for part in self.parse(buffer):
            yield [part]


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "output_parsers", "list"]

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz`"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern = r"\d+\.\s([^\n]+)"

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return re.findall(self.pattern, text)

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call."""
        return re.finditer(self.pattern, text)

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a markdown list."""

    pattern = r"^\s*[-*]\s([^\n]+)$"

    def get_format_instructions(self) -> str:
        return "Your response should be a markdown list, " "eg: `- foo\n- bar\n- baz`"

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return re.findall(self.pattern, text, re.MULTILINE)

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call."""
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"
