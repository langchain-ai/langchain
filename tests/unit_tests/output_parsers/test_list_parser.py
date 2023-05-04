from langchain.output_parsers.list import CommaSeparatedListOutputParser


def test_single_item() -> None:
    """Test that a string with a single item is parsed to a list with that item."""
    parser = CommaSeparatedListOutputParser()
    assert parser.parse("foo") == ["foo"]


def test_multiple_items() -> None:
    """Test that a string with multiple comma-separated items is parsed to a list."""
    parser = CommaSeparatedListOutputParser()
    assert parser.parse("foo, bar, baz") == ["foo", "bar", "baz"]
