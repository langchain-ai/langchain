from langchain_core.output_parsers import StrOutputParser


def test_lc_namespace() -> None:
    assert StrOutputParser.get_lc_namespace() == [
        "langchain",
        "schema",
        "output_parser",
    ]
