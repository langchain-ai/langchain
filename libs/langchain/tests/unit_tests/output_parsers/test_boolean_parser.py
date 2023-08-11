from langchain.output_parsers.boolean import BooleanOutputParser


def test_boolean_output_parser_parse() -> None:
    parser = BooleanOutputParser()

    # Test valid input
    result = parser.parse("YES")
    assert result is True

    # Test valid input
    result = parser.parse("NO")
    assert result is False

    # Test valid input
    result = parser.parse("yes")
    assert result is True

    # Test valid input
    result = parser.parse("no")
    assert result is False

    # Test invalid input
    try:
        parser.parse("INVALID")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
