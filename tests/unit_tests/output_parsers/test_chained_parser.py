from langchain.output_parsers.remove_quotes import RemoveQuotesOutputParser
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.chained import ChainedOutputParser


def test_boolean_output_parser_parse() -> None:
    remove_quotes_parser = RemoveQuotesOutputParser()
    boolean_parser = BooleanOutputParser()
    chained_parser = ChainedOutputParser(parsers=[remove_quotes_parser, boolean_parser])

    # Test valid input
    result = chained_parser.parse("'YES'")
    assert result is True

    # Test valid input
    result = chained_parser.parse("NO")
    assert result is False

    # Test invalid input
    try:
        chained_parser.parse("INVALID")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
