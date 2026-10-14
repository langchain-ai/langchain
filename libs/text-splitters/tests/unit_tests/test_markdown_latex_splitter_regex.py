"""Regression tests for #41184.

`MarkdownTextSplitter` and `LatexTextSplitter` take their separators from
`RecursiveCharacterTextSplitter.get_separators_for_language(...)`, which returns
regex patterns (for Markdown, heading separators such as a run of hash marks).
Those separators must be treated as regexes, otherwise they are escaped and
never match real text.
"""

from __future__ import annotations

from langchain_text_splitters import (
    Language,
    LatexTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)


def test_markdown_text_splitter_splits_on_headings() -> None:
    """`MarkdownTextSplitter` should treat its separators as regexes."""
    text = "# T\nintro\n## B\ntext b\n### C\ntext c\n"
    splitter = MarkdownTextSplitter(chunk_size=20, chunk_overlap=0)
    assert splitter._is_separator_regex is True
    chunks = splitter.split_text(text)
    assert chunks == ["# T\nintro", "## B\ntext b", "### C\ntext c"]
    expected = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=20, chunk_overlap=0
    ).split_text(text)
    assert chunks == expected


def test_latex_text_splitter_splits_on_sections() -> None:
    """`LatexTextSplitter` should treat its separators as regexes."""
    text = "\\section{A} aa bb\n\\section{B} cc dd\n\\section{C} ee ff"
    splitter = LatexTextSplitter(chunk_size=24, chunk_overlap=0)
    assert splitter._is_separator_regex is True
    chunks = splitter.split_text(text)
    assert chunks == [
        "\\section{A} aa bb",
        "\\section{B} cc dd",
        "\\section{C} ee ff",
    ]
    expected = RecursiveCharacterTextSplitter.from_language(
        Language.LATEX, chunk_size=24, chunk_overlap=0
    ).split_text(text)
    assert chunks == expected
