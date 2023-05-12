from langchain.output_parsers.smart_boolean import SmartBooleanOutputParser


def test_smart_boolean_output_parser_parse() -> None:
    parser = SmartBooleanOutputParser()

    # Test valid input
    result = parser.parse("YES")
    assert result is True

    # Test valid input
    result = parser.parse("NO")
    assert result is False

    # Test valid input
    result = parser.parse("Yes, that's correct. Consider 4 reasons why....")
    assert result is True

    # Test invalid input
    try:
        parser.parse("INVALID")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
