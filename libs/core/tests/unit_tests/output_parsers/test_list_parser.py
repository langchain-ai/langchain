from typing import AsyncIterator, Iterable, List, TypeVar, cast

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


def test_multiple_items() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
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
        expectedlist = [[a] for a in cast(List[str], expected)]
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
        "Your response should be a numbered - not a list item - list with each item on a new line."  # noqa: E501
        "For example: \n- foo\n- bar\n- baz"
    )

    text2 = "Items:\n- apple\n     - banana\n- cherry"

    text3 = "No items in the list."

    for text, expected in [
        (text1, ["foo", "bar", "baz"]),
        (text2, ["apple", "banana", "cherry"]),
        (text3, []),
    ]:
        expectedlist = [[a] for a in cast(List[str], expected)]
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
        expectedlist = [[a] for a in cast(List[str], expected)]
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
        expectedlist = [[a] for a in cast(List[str], expected)]
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
