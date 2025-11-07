"""Markdown text splitters."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.documents import Document

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)


class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        return_each_line: bool = False,  # noqa: FBT001,FBT002
        strip_headers: bool = True,  # noqa: FBT001,FBT002
        custom_header_patterns: dict[str, int] | None = None,
    ) -> None:
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
            custom_header_patterns: Optional dict mapping header patterns to their
                levels. For example: {"**": 1, "***": 2} to treat **Header** as
                level 1 and ***Header*** as level 2 headers.
        """
        # Output line-by-line or aggregated into chunks w/ common headers
        self.return_each_line = return_each_line
        # Given the headers we want to split on,
        # (e.g., "#, ##, etc") order by length
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        # Strip headers split headers from the content of the chunk
        self.strip_headers = strip_headers
        # Custom header patterns with their levels
        self.custom_header_patterns = custom_header_patterns or {}

    def _is_custom_header(self, line: str, sep: str) -> bool:
        """Check if line matches a custom header pattern.

        Args:
            line: The line to check
            sep: The separator pattern to match

        Returns:
            True if the line matches the custom pattern format
        """
        if sep not in self.custom_header_patterns:
            return False

        # Escape special regex characters in the separator
        escaped_sep = re.escape(sep)
        # Create regex pattern to match exactly one separator at start and end
        # with content in between
        pattern = (
            f"^{escaped_sep}(?!{escaped_sep})(.+?)(?<!{escaped_sep}){escaped_sep}$"
        )

        match = re.match(pattern, line)
        if match:
            # Extract the content between the patterns
            content = match.group(1).strip()
            # Valid header if there's actual content (not just whitespace or separators)
            # Check that content doesn't consist only of separator characters
            if content and not all(c in sep for c in content.replace(" ", "")):
                return True
        return False

    def aggregate_lines_to_chunks(self, lines: list[LineType]) -> list[Document]:
        """Combine lines with common metadata into chunks.

        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: list[LineType] = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            elif (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] != line["metadata"]
                # may be issues if other metadata is present
                and len(aggregated_chunks[-1]["metadata"]) < len(line["metadata"])
                and aggregated_chunks[-1]["content"].split("\n")[-1][0] == "#"
                and not self.strip_headers
            ):
                # If the last line in the aggregated list
                # has different metadata as the current line,
                # and has shallower header level than the current line,
                # and the last line is a header,
                # and we are not stripping headers,
                # append the current content to the last line's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
                # and update the last line's metadata
                aggregated_chunks[-1]["metadata"] = line["metadata"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text(self, text: str) -> list[Document]:
        """Split markdown file.

        Args:
            text: Markdown file
        """
        # Split the input text by newline character ("\n").
        lines = text.split("\n")
        # Final output
        lines_with_metadata: list[LineType] = []
        # Content and metadata of the chunk currently being processed
        current_content: list[str] = []
        current_metadata: dict[str, str] = {}
        # Keep track of the nested header structure
        header_stack: list[HeaderType] = []
        initial_metadata: dict[str, str] = {}

        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
            # Remove all non-printable characters from the string, keeping only visible
            # text.
            stripped_line = "".join(filter(str.isprintable, stripped_line))
            if not in_code_block:
                # Exclude inline code spans
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            elif stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ""

            if in_code_block:
                current_content.append(stripped_line)
                continue

            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                is_standard_header = stripped_line.startswith(sep) and (
                    # Header with no text OR header is followed by space
                    # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                )
                is_custom_header = self._is_custom_header(stripped_line, sep)

                # Check if line matches either standard or custom header pattern
                if is_standard_header or is_custom_header:
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        if sep in self.custom_header_patterns:
                            current_header_level = self.custom_header_patterns[sep]
                        else:
                            current_header_level = sep.count("#")

                        # Pop out headers of lower or same level from the stack
                        while (
                            header_stack
                            and header_stack[-1]["level"] >= current_header_level
                        ):
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        # Extract header text based on header type
                        if is_custom_header:
                            # For custom headers like **Header**, extract text
                            # between patterns
                            header_text = stripped_line[len(sep) : -len(sep)].strip()
                        else:
                            # For standard headers like # Header, extract text
                            # after the separator
                            header_text = stripped_line[len(sep) :].strip()

                        header: HeaderType = {
                            "level": current_header_level,
                            "name": name,
                            "data": header_text,
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    if not self.strip_headers:
                        current_content.append(stripped_line)

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {
                    "content": "\n".join(current_content),
                    "metadata": current_metadata,
                }
            )

        # lines_with_metadata has each line with associated header metadata
        # aggregate these into chunks based on common metadata
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in lines_with_metadata
        ]


class LineType(TypedDict):
    """Line type as typed dict."""

    metadata: dict[str, str]
    content: str


class HeaderType(TypedDict):
    """Header type as typed dict."""

    level: int
    name: str
    data: str


class ExperimentalMarkdownSyntaxTextSplitter:
    """An experimental text splitter for handling Markdown syntax.

    This splitter aims to retain the exact whitespace of the original text while
    extracting structured metadata, such as headers. It is a re-implementation of the
    MarkdownHeaderTextSplitter with notable changes to the approach and
    additional features.

    Key Features:

    * Retains the original whitespace and formatting of the Markdown text.
    * Extracts headers, code blocks, and horizontal rules as metadata.
    * Splits out code blocks and includes the language in the "Code" metadata key.
    * Splits text on horizontal rules (`---`) as well.
    * Defaults to sensible splitting behavior, which can be overridden using the
        `headers_to_split_on` parameter.

    Example:
    ```python
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    chunks = splitter.split(text)
    for chunk in chunks:
        print(chunk)
    ```

    This class is currently experimental and subject to change based on feedback and
    further development.
    """

    DEFAULT_HEADER_KEYS = {
        "#": "Header 1",
        "##": "Header 2",
        "###": "Header 3",
        "####": "Header 4",
        "#####": "Header 5",
        "######": "Header 6",
    }

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        return_each_line: bool = False,  # noqa: FBT001,FBT002
        strip_headers: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Initialize the text splitter with header splitting and formatting options.

        This constructor sets up the required configuration for splitting text into
        chunks based on specified headers and formatting preferences.

        Args:
            headers_to_split_on (Union[list[tuple[str, str]], None]):
                A list of tuples, where each tuple contains a header tag (e.g., "h1")
                and its corresponding metadata key. If `None`, default headers are used.
            return_each_line (bool):
                Whether to return each line as an individual chunk.
                Defaults to `False`, which aggregates lines into larger chunks.
            strip_headers (bool):
                Whether to exclude headers from the resulting chunks.
        """
        self.chunks: list[Document] = []
        self.current_chunk = Document(page_content="")
        self.current_header_stack: list[tuple[int, str]] = []
        self.strip_headers = strip_headers
        if headers_to_split_on:
            self.splittable_headers = dict(headers_to_split_on)
        else:
            self.splittable_headers = self.DEFAULT_HEADER_KEYS

        self.return_each_line = return_each_line

    def split_text(self, text: str) -> list[Document]:
        """Split the input text into structured chunks.

        This method processes the input text line by line, identifying and handling
        specific patterns such as headers, code blocks, and horizontal rules to
        split it into structured chunks based on headers, code blocks, and
        horizontal rules.

        Args:
            text: The input text to be split into chunks.

        Returns:
            A list of `Document` objects representing the structured
            chunks of the input text. If `return_each_line` is enabled, each line
            is returned as a separate `Document`.
        """
        # Reset the state for each new file processed
        self.chunks.clear()
        self.current_chunk = Document(page_content="")
        self.current_header_stack.clear()

        raw_lines = text.splitlines(keepends=True)

        while raw_lines:
            raw_line = raw_lines.pop(0)
            header_match = self._match_header(raw_line)
            code_match = self._match_code(raw_line)
            horz_match = self._match_horz(raw_line)
            if header_match:
                self._complete_chunk_doc()

                if not self.strip_headers:
                    self.current_chunk.page_content += raw_line

                # add the header to the stack
                header_depth = len(header_match.group(1))
                header_text = header_match.group(2)
                self._resolve_header_stack(header_depth, header_text)
            elif code_match:
                self._complete_chunk_doc()
                self.current_chunk.page_content = self._resolve_code_chunk(
                    raw_line, raw_lines
                )
                self.current_chunk.metadata["Code"] = code_match.group(1)
                self._complete_chunk_doc()
            elif horz_match:
                self._complete_chunk_doc()
            else:
                self.current_chunk.page_content += raw_line

        self._complete_chunk_doc()
        # I don't see why `return_each_line` is a necessary feature of this splitter.
        # It's easy enough to do outside of the class and the caller can have more
        # control over it.
        if self.return_each_line:
            return [
                Document(page_content=line, metadata=chunk.metadata)
                for chunk in self.chunks
                for line in chunk.page_content.splitlines()
                if line and not line.isspace()
            ]
        return self.chunks

    def _resolve_header_stack(self, header_depth: int, header_text: str) -> None:
        for i, (depth, _) in enumerate(self.current_header_stack):
            if depth >= header_depth:
                # Truncate everything from this level onward
                self.current_header_stack = self.current_header_stack[:i]
                break
        self.current_header_stack.append((header_depth, header_text))

    def _resolve_code_chunk(self, current_line: str, raw_lines: list[str]) -> str:
        chunk = current_line
        while raw_lines:
            raw_line = raw_lines.pop(0)
            chunk += raw_line
            if self._match_code(raw_line):
                return chunk
        return ""

    def _complete_chunk_doc(self) -> None:
        chunk_content = self.current_chunk.page_content
        # Discard any empty documents
        if chunk_content and not chunk_content.isspace():
            # Apply the header stack as metadata
            for depth, value in self.current_header_stack:
                header_key = self.splittable_headers.get("#" * depth)
                self.current_chunk.metadata[header_key] = value
            self.chunks.append(self.current_chunk)
        # Reset the current chunk
        self.current_chunk = Document(page_content="")

    # Match methods
    def _match_header(self, line: str) -> re.Match[str] | None:
        match = re.match(r"^(#{1,6}) (.*)", line)
        # Only matches on the configured headers
        if match and match.group(1) in self.splittable_headers:
            return match
        return None

    def _match_code(self, line: str) -> re.Match[str] | None:
        matches = [re.match(rule, line) for rule in [r"^```(.*)", r"^~~~(.*)"]]
        return next((match for match in matches if match), None)

    def _match_horz(self, line: str) -> re.Match[str] | None:
        matches = [
            re.match(rule, line) for rule in [r"^\*\*\*+\n", r"^---+\n", r"^___+\n"]
        ]
        return next((match for match in matches if match), None)
