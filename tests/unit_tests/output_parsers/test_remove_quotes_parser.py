import itertools
from langchain.output_parsers.remove_quotes import RemoveQuotesOutputParser


def test_remove_quotes_output_parser_parse() -> None:
    parser = RemoveQuotesOutputParser()

    # Test all combinations of all quotes
    quote_combinations = [
        combo
        for i in range(len(parser.quotes) + 1)
        for combo in itertools.combinations(parser.quotes, i)
    ]
    for quote_combo in quote_combinations:
        input = "hello"
        for left, right in quote_combo:
            input = f"{left}{input}{right}"
        result = parser.parse(input)
        assert result == "hello"

    # Test unbaleanced quotes
    # (they should be considered part of the response rather than removed)
    for left, right in parser.quotes:
        input = f"{left}hello"
        result = parser.parse(input)
        assert result == input

        input = f"hello{right}"
        result = parser.parse(input)
        assert result == input
