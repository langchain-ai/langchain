from langchain.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)


def test_single_item() -> None:
    """Test that a string with a single item is parsed to a list with that item."""
    parser = CommaSeparatedListOutputParser()
    assert parser.parse("foo") == ["foo"]


def test_multiple_items() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
    parser = CommaSeparatedListOutputParser()
    assert parser.parse("foo, bar, baz") == ["foo", "bar", "baz"]


def test_numbered_list() -> None:
    parser = NumberedListOutputParser()
    text1 = (
        "Your response should be a numbered list with each item on a new line. "
        "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
    )

    text2 = "Items:\n\n1. apple\n\n2. banana\n\n3. cherry"

    text3 = "No items in the list."

    assert parser.parse(text1) == ["foo", "bar", "baz"]
    assert parser.parse(text2) == ["apple", "banana", "cherry"]
    assert parser.parse(text3) == []


def test_markdown_list() -> None:
    parser = MarkdownListOutputParser()
    text1 = (
        "Your response should be a numbered list with each item on a new line."
        "For example: \n- foo\n- bar\n- baz"
    )

    text2 = "Items:\n- apple\n- banana\n- cherry"

    text3 = "No items in the list."

    assert parser.parse(text1) == ["foo", "bar", "baz"]
    assert parser.parse(text2) == ["apple", "banana", "cherry"]
    assert parser.parse(text3) == []
