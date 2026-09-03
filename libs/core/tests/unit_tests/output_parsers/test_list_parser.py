from collections.abc import AsyncIterator, Iterable
from typing import TypeVar

import pytest

from langchain_core.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)

T = TypeVar("T")


async def aiter_from_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    """Convert a synchronous iterable to an asynchronous iterator."""
    for item in iterable:
        yield item


def add(generator):
    """Helper to consume a synchronous transform generator."""
    res = None
    for chunk in generator:
        if res is None:
            res = chunk
        else:
            res += chunk
    return res


async def aadd(generator):
    """Helper to consume an asynchronous transform generator."""
    res = None
    async for chunk in generator:
        if res is None:
            res = chunk
        else:
            res += chunk
    return res


def test_single_item() -> None:
    parser = CommaSeparatedListOutputParser()
    text = "foo"
    expected = ["foo"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [expected]


def test_multiple_items_with_spaces() -> None:
    parser = CommaSeparatedListOutputParser()
    text = "foo, bar, baz"
    expected = ["foo", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [expected]


def test_multiple_items() -> None:
    parser = CommaSeparatedListOutputParser()
    text = "foo,bar,baz"
    expected = ["foo", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [expected]


def test_multiple_items_with_comma() -> None:
    parser = CommaSeparatedListOutputParser()
    text = '"foo, foo2",bar,baz'
    expected = ["foo, foo2", "bar", "baz"]

    assert parser.parse(text) == expected
    assert add(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text)) == [expected]


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
        assert parser.parse(text) == expected
        assert add(parser.transform(t for t in text)) == expected
        assert list(parser.transform(t for t in text)) == [expected]


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
        assert parser.parse(text) == expected
        assert add(parser.transform(t for t in text)) == expected
        assert list(parser.transform(t for t in text)) == [expected]


@pytest.mark.asyncio
async def test_single_item_async() -> None:
    parser = CommaSeparatedListOutputParser()
    text = "foo"
    expected = ["foo"]

    assert await parser.aparse(text) == expected
    assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
    assert [a async for a in parser.atransform(aiter_from_iter(t for t in text))] == [
        expected
    ]


@pytest.mark.asyncio
async def test_multiple_items_async() -> None:
    parser = CommaSeparatedListOutputParser()
    text = "foo, bar, baz"
    expected = ["foo", "bar", "baz"]

    assert await parser.aparse(text) == expected
    assert await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
    assert [a async for a in parser.atransform(aiter_from_iter(t for t in text))] == [
        expected
    ]


@pytest.mark.asyncio
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
        assert await parser.aparse(text) == expected
        assert (
            await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
        )
        assert [
            a async for a in parser.atransform(aiter_from_iter(t for t in text))
        ] == [expected]


@pytest.mark.asyncio
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
        assert await parser.aparse(text) == expected
        assert (
            await aadd(parser.atransform(aiter_from_iter(t for t in text))) == expected
        )
        assert [
            a async for a in parser.atransform(aiter_from_iter(t for t in text))
        ] == [expected]
