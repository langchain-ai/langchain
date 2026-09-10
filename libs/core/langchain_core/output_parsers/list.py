"""Parsers for list output."""

from __future__ import annotations

import csv
import re
from abc import abstractmethod
from collections import deque
from io import StringIO
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

T = TypeVar("T")


def droplastn(
    iter: Iterator[T],  # noqa: A002
    n: int,
) -> Iterator[T]:
    """Drop the last `n` elements of an iterator.

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


def _decode_csv_field(field: str) -> str:
    """Decode a single raw CSV field using the same dialect as `parse`.

    Args:
        field: The raw text of one CSV field, without its trailing separator.

    Returns:
        The decoded field value.
    """
    try:
        reader = csv.reader(
            StringIO(field), quotechar='"', delimiter=",", skipinitialspace=True
        )
        row = next(reader, None)
    except csv.Error:
        return field.strip()
    if not row:
        return ""
    return row[0]


def _split_complete_csv_fields(text: str) -> tuple[list[str], int]:
    """Split completed CSV fields out of a raw, possibly partial CSV stream.

    Scans `text` with quote awareness and returns the decoded fields that are
    already terminated by a separator (a `,` or a record boundary outside
    quotes), together with the index in `text` where the raw remainder (the
    trailing field, which may still be incomplete) starts. Keeping the
    remainder raw preserves quote state across streaming chunks, unlike
    buffering the already-decoded last field.

    Args:
        text: The raw CSV text accumulated so far.

    Returns:
        A tuple of the decoded complete fields and the start index of the raw
        remainder (`len(text)` when every field is complete).
    """
    fields: list[str] = []
    start = 0
    i = 0
    n = len(text)
    in_quotes = False
    while i < n:
        char = text[i]
        if in_quotes:
            if char == '"':
                if i + 1 < n and text[i + 1] == '"':
                    # Escaped quote inside a quoted field.
                    i += 2
                    continue
                in_quotes = False
            i += 1
        elif char == '"':
            in_quotes = True
            i += 1
        elif char == ",":
            fields.append(_decode_csv_field(text[start:i]))
            start = i + 1
            i += 1
        elif char in ("\r", "\n"):
            if char == "\r" and i + 1 == n:
                # A trailing `\r` may be the first half of a `\r\n` pair that
                # arrives with the next chunk; do not split on it yet.
                break
            fields.append(_decode_csv_field(text[start:i]))
            start = i + 1
            if char == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 2
            else:
                i += 1
        else:
            i += 1
    return fields, start


class ListOutputParser(BaseTransformOutputParser[list[str]]):
    """Parse the output of a model to a list."""

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

    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Yields:
            A match object for each part of the output.
        """
        raise NotImplementedError

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[list[str]]:
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                buffer += chunk
            # Parse buffer into a list of parts
            try:
                done_idx = 0
                # Yield only complete parts
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # Yield only complete parts
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # Yield the last part
        for part in self.parse(buffer):
            yield [part]

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[list[str]]:
        buffer = ""
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                buffer += chunk
            # Parse buffer into a list of parts
            try:
                done_idx = 0
                # Yield only complete parts
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # Yield only complete parts
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # Yield the last part
        for part in self.parse(buffer):
            yield [part]


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of a model to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "output_parsers", "list"]`
        """
        return ["langchain", "output_parsers", "list"]

    @override
    def get_format_instructions(self) -> str:
        """Return the format instructions for the comma-separated list output."""
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz` or `foo,bar,baz`"
        )

    @override
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
            # Keep old logic for backup
            return [part.strip() for part in text.split(",")]

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[list[str]]:
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                buffer += chunk
            # Yield only fields already terminated by a separator, keeping the
            # raw remainder (quote state intact) as the next buffer
            fields, done_idx = _split_complete_csv_fields(buffer)
            if done_idx:
                for field in fields:
                    yield [field]
                buffer = buffer[done_idx:]
        # Yield the last part
        for part in self.parse(buffer):
            yield [part]

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[list[str]]:
        buffer = ""
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                buffer += chunk
            # Yield only fields already terminated by a separator, keeping the
            # raw remainder (quote state intact) as the next buffer
            fields, done_idx = _split_complete_csv_fields(buffer)
            if done_idx:
                for field in fields:
                    yield [field]
                buffer = buffer[done_idx:]
        # Yield the last part
        for part in self.parse(buffer):
            yield [part]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list."""

    pattern: str = r"\d+\.\s([^\n]+)"
    """The pattern to match a numbered list item."""

    @override
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

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text)

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a Markdown list."""

    pattern: str = r"^\s*[-*]\s([^\n]+)$"
    """The pattern to match a Markdown list item."""

    @override
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

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match[str]]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"
