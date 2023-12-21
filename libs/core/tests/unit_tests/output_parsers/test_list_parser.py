from langchain_core.output_parsers.list import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)


def test_single_item() -> None:
    """Test that a string with a single item is parsed to a list with that item."""
    parser = CommaSeparatedListOutputParser()
    text = "foo"
    expected = ["foo"]

    assert parser.parse(text) == expected
    assert list(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == expected
    assert (
        list(
            parser.transform(
                " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
            )
        )
        == expected
    )
    assert list(parser.transform([text])) == expected


def test_multiple_items() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
    parser = CommaSeparatedListOutputParser()
    text = "foo, bar, baz"
    expected = ["foo", "bar", "baz"]

    assert parser.parse(text) == expected
    assert list(parser.transform(t for t in text)) == expected
    assert list(parser.transform(t for t in text.splitlines(keepends=True))) == expected
    assert (
        list(
            parser.transform(
                " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
            )
        )
        == expected
    )
    assert list(parser.transform([text])) == expected


def test_numbered_list() -> None:
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
        assert parser.parse(text) == expected
        assert list(parser.transform(t for t in text)) == expected
        assert (
            list(parser.transform(t for t in text.splitlines(keepends=True)))
            == expected
        )
        assert (
            list(
                parser.transform(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
            == expected
        )
        assert list(parser.transform([text])) == expected


def test_markdown_list() -> None:
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
        assert parser.parse(text) == expected
        assert list(parser.transform(t for t in text)) == expected
        assert (
            list(parser.transform(t for t in text.splitlines(keepends=True)))
            == expected
        )
        assert (
            list(
                parser.transform(
                    " " + t if i > 0 else t for i, t in enumerate(text.split(" "))
                )
            )
            == expected
        )
        assert list(parser.transform([text])) == expected
