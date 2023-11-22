from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
)


def test_lc_namespace() -> None:
    assert CommaSeparatedListOutputParser.get_lc_namespace() == [
        "langchain",
        "output_parsers",
        "list",
    ]
    assert NumberedListOutputParser.get_lc_namespace() == [
        "langchain",
        "output_parsers",
        "list",
    ]
    assert MarkdownListOutputParser.get_lc_namespace() == [
        "langchain",
        "output_parsers",
        "list",
    ]
