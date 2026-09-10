from collections.abc import AsyncIterator, Iterable
from typing import TypeVar

from langchain_core.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)
from langchain_core.runnables.utils import aadd, add


def test_single_item() -> None:
    """Test that a string with a single item is parsed to a list with that item."""
    parser = CommaSeparatedListOutputParser()
    text = "foo"
    expected = ["foo"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [[a] for a in expected]
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == [
        [a] for a in expected
    ]
    assert list(
        parser.transform(" " + t if i > 0 else t for i, t in enumerate(text.split(" ")))
    ) == [[a] for a in expected]
    assert list(parser.transform(iter([text]))) == [[a] for a in expected]


def test_multiple_items_with_spaces() -> None:
    """Test multiple items with spaces.

    Test that a string with multiple comma-separated items
    with spaces is parsed to a list.
    """
    parser = CommaSeparatedListOutputParser()
    text = "foo, bar, baz"
    expected = ["foo", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [[a] for a in expected]
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == [
        [a] for a in expected
    ]
    assert list(
        parser.transform(" " + t if i > 0 else t for i, t in enumerate(text.split(" ")))
    ) == [[a] for a in expected]
    assert list(parser.transform(iter([text]))) == [[a] for a in expected]


def test_multiple_items() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
    parser = CommaSeparatedListOutputParser()
    text = "foo,bar,baz"
    expected = ["foo", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [[a] for a in expected]
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == [
        [a] for a in expected
    ]
    assert list(
        parser.transform(" " + t if i > 0 else t for i, t in enumerate(text.split(" ")))
    ) == [[a] for a in expected]
    assert list(parser.transform(iter([text]))) == [[a] for a in expected]


def test_multiple_items_with_comma() -> None:
    """Test multiple items with a comma.

    Test that a string with multiple comma-separated items with 1 item containing a
    comma is parsed to a list.
    """
    parser = CommaSeparatedListOutputParser()
    text = '"foo, foo2",bar,baz'
    expected = ["foo, foo2", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [[a] for a in expected]
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == [
        [a] for a in expected
    ]
    assert list(
        parser.transform(" " + t if i > 0 else t for i, t in enumerate(text.split(" ")))
    ) == [[a] for a in expected]
    assert list(parser.transform(iter([text]))) == [[a] for a in expected]


def test_numbered_list() -> None:
    parser = NumberedListOutputParser()
    text1 = (
        "Your response should be a numbered list with each item on a new line. "
        "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
    )

    text2 = "Items:\n\n1. apple\n\n    2. banana\n\n3. cherry"

    text3 = "No items in the list."

    for text, expected in [
        (text1, ["foo", "bar", "baz"]),
        (text2, ["apple", "banana", "cherry"]),
        (text3, []),
    ]:
        expectedlist = [[a] for a in expected]
        assert parser.parse(text) == expected
        assert add(parser.transform(t for t in text)) == (expected or None)
        assert list(parser.transform(t for t in text)) == expectedlist
        assert (
            list(parser.transform(t for t in text.splitlines(keepends=True)))
            == expectedlist
        )
        assert (
            list(
                parser.transform(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
            == expectedlist
        )
        assert list(parser.transform(iter([text]))) == expectedlist


def test_markdown_list() -> None:
    parser = MarkdownListOutputParser()
    text1 = (
        "Your response should be a numbered - not a list item - "
        "list with each item on a new line."
        "For example: \n- foo\n- bar\n- baz"
    )

    text2 = "Items:\n- apple\n     - banana\n- cherry"

    text3 = "No items in the list."

    for text, expected in [
        (text1, ["foo", "bar", "baz"]),
        (text2, ["apple", "banana", "cherry"]),
        (text3, []),
    ]:
        expectedlist = [[a] for a in expected]
        assert parser.parse(text) == expected
        assert add(parser.transform(t for t in text)) == (expected or None)
        assert list(parser.transform(t for t in text)) == expectedlist
        assert (
            list(parser.transform(t for t in text.splitlines(keepends=True)))
            == expectedlist
        )
        assert (
            list(
                parser.transform(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
            == expectedlist
        )
        assert list(parser.transform(iter([text]))) == expectedlist


T = TypeVar("T")


async def aiter_from_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    for item in iterable:
        yield item


async def test_single_item_async() -> None:
    """Test that a string with a single item is parsed to a list with that item."""
    parser = CommaSeparatedListOutputParser()
    text = "foo"
    expected = ["foo"]

    assert await parser.aparse(text) == expected
    assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
    assert [a async for a in parser.atransform(aiter_from_iter(t for t in text))] == [
        [a] for a in expected
    ]
    assert [
        a
        async for a in parser.atransform(
            aiter_from_iter(t for t in text.splitlines(keepends=True))
        )
    ] == [[a] for a in expected]
    assert [
        a
        async for a in parser.atransform(
            aiter_from_iter(
                " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
            )
        )
    ] == [[a] for a in expected]
    assert [a async for a in parser.atransform(aiter_from_iter([text]))] == [
        [a] for a in expected
    ]


async def test_multiple_items_async() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
    parser = CommaSeparatedListOutputParser()
    text = "foo, bar, baz"
    expected = ["foo", "bar", "baz"]

    assert await parser.aparse(text) == expected
    assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
    assert [a async for a in parser.atransform(aiter_from_iter(t for t in text))] == [
        [a] for a in expected
    ]
    assert [
        a
        async for a in parser.atransform(
            aiter_from_iter(t for t in text.splitlines(keepends=True))
        )
    ] == [[a] for a in expected]
    assert [
        a
        async for a in parser.atransform(
            aiter_from_iter(
                " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
            )
        )
    ] == [[a] for a in expected]
    assert [a async for a in parser.atransform(aiter_from_iter([text]))] == [
        [a] for a in expected
    ]


async def test_numbered_list_async() -> None:
    parser = NumberedListOutputParser()
    text1 = (
        "Your response should be a numbered list with each item on a new line. "
        "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
    )

    text2 = "Items:\n\n1. apple\n\n2. banana\n\n3. cherry"

    text3 = "No items in the list."

    for text, expected in [
        (text1, ["foo", "bar", "baz"]),
        (text2, ["apple", "banana", "cherry"]),
        (text3, []),
    ]:
        expectedlist = [[a] for a in expected]
        assert await parser.aparse(text) == expected
        assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == (
            expected or None
        )
        assert [
            a async for a in parser.atransform(aiter_from_iter(t for t in text))
        ] == expectedlist
        assert [
            a
            async for a in parser.atransform(
                aiter_from_iter(t for t in text.splitlines(keepends=True))
            )
        ] == expectedlist
        assert [
            a
            async for a in parser.atransform(
                aiter_from_iter(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
        ] == expectedlist
        assert [
            a async for a in parser.atransform(aiter_from_iter([text]))
        ] == expectedlist


async def test_markdown_list_async() -> None:
    parser = MarkdownListOutputParser()
    text1 = (
        "Your response should be a numbered list with each item on a new line."
        "For example: \n- foo\n- bar\n- baz"
    )

    text2 = "Items:\n- apple\n- banana\n- cherry"

    text3 = "No items in the list."

    for text, expected in [
        (text1, ["foo", "bar", "baz"]),
        (text2, ["apple", "banana", "cherry"]),
        (text3, []),
    ]:
        expectedlist = [[a] for a in expected]
        assert await parser.aparse(text) == expected
        assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == (
            expected or None
        )
        assert [
            a async for a in parser.atransform(aiter_from_iter(t for t in text))
        ] == expectedlist
        assert [
            a
            async for a in parser.atransform(
                aiter_from_iter(t for t in text.splitlines(keepends=True))
            )
        ] == expectedlist
        assert [
            a
            async for a in parser.atransform(
                aiter_from_iter(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
        ] == expectedlist
        assert [
            a async for a in parser.atransform(aiter_from_iter([text]))
        ] == expectedlist


def test_streaming_quoted_field_across_chunk_boundary() -> None:
    """Streaming must produce the same fields as non-streaming for a quoted field
    that contains a comma and is split across chunk boundaries (#40360).

    Before the fix, the streaming fallback stored the decoded last part (e.g.
    ``beta,``) as the next buffer instead of the raw unprocessed text (e.g.
    `` "beta,``), so the opening quote was lost and subsequent CSV parsing
    produced wrong field boundaries.
    """
    parser = CommaSeparatedListOutputParser()

    # Quoted field "beta, gamma" split across two chunks
    chunks = ['alpha, "beta,', ' gamma", delta']
    full_text = "".join(chunks)

    # Establish the correct parse result via non-streaming baseline
    expected = parser.parse(full_text)
    assert expected == ["alpha", "beta, gamma", "delta"]
    expected_streamed = [[item] for item in expected]

    assert list(parser.transform(iter(chunks))) == expected_streamed
    assert add(parser.transform(iter(chunks))) == expected

    # Doubled quote inside a quoted field, split across chunks
    chunks2 = ['x, "it''s a test,', ' value", z']
    full_text2 = "".join(chunks2)
    expected2 = parser.parse(full_text2)
    assert list(parser.transform(iter(chunks2))) == [[item] for item in expected2]

    # Field spanning three chunks
    chunks3 = ['"start,', ' middle,', ' end", last']
    full_text3 = "".join(chunks3)
    expected3 = parser.parse(full_text3)
    assert expected3 == ["start, middle, end", "last"]
    assert list(parser.transform(iter(chunks3))) == [[item] for item in expected3]


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_streaming_quoted_field_across_chunk_boundary_async() -> None:
    """Async streaming must also correctly handle quoted fields split across chunks."""
    parser = CommaSeparatedListOutputParser()

    chunks = ['alpha, "beta,', ' gamma", delta']
    full_text = "".join(chunks)
    expected = parser.parse(full_text)
    assert expected == ["alpha", "beta, gamma", "delta"]
    expected_streamed = [[item] for item in expected]

    result = [a async for a in parser.atransform(aiter_from_iter(iter(chunks)))]
    assert result == expected_streamed
    assert await aadd(parser.atransform(aiter_from_iter(iter(chunks)))) == expected
