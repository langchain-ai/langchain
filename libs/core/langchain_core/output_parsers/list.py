from __future__ import annotations

import csv
import re
from abc import abstractmethod
from collections import deque
from collections.abc import AsyncIterator, Iterator
from io import StringIO
from typing import Optional as Optional
from typing import TypeVar, Union

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

T = TypeVar("T")


def droplastn(iter: Iterator[T], n: int) -> Iterator[T]:
    """Drop the last n elements of an iterator.

    Args:
        iter: The iterator to drop elements from.
        n: The number of elements to drop.

    Yields:
        The elements of the iterator, except the last n elements.
    """
    buffer: deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()


class ListOutputParser(BaseTransformOutputParser[list[str]]):
    """Parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Yields:
            A match object for each part of the output.
        """
        raise NotImplementedError

    def _transform(
        self, input: Iterator[Union[str, BaseMessage]]
    ) -> Iterator[list[str]]:
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
    ) -> AsyncIterator[list[str]]:
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


ListOutputParser.model_rebuild()


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the langchain object is serializable.

        Returns True.
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            A list of strings.
            Default is ["langchain", "output_parsers", "list"].
        """
        return ["langchain", "output_parsers", "list"]

    def get_format_instructions(self) -> str:
        """Return the format instructions for the comma-separated list output."""
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz` or `foo,bar,baz`"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        try:
            reader = csv.reader(
                StringIO(text), quotechar='"', delimiter=",", skipinitialspace=True
            )
            return [item for sublist in reader for item in sublist]
        except csv.Error:
            # keep old logic for backup
            return [part.strip() for part in text.split(",")]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern: str = r"\d+\.\s([^\n]+)"
    """The pattern to match a numbered list item."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        return re.findall(self.pattern, text)

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Yields:
            A match object for each part of the output.
        """
        return re.finditer(self.pattern, text)

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a Markdown list."""

    pattern: str = r"^\s*[-*]\s([^\n]+)$"
    """The pattern to match a Markdown list item."""

    def get_format_instructions(self) -> str:
        """Return the format instructions for the Markdown list output."""
        return "Your response should be a markdown list, eg: `- foo\n- bar\n- baz`"

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        """
        return re.findall(self.pattern, text, re.MULTILINE)

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Yields:
            A match object for each part of the output.
        """
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"
