from langchain.output_parsers.choice import ChoiceOutputParser
from langchain.schema import OutputParserException


def test_choice_output_parser_parse() -> None:
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    parser = ChoiceOutputParser(options=[RED, GREEN, BLUE])

    # Test valid inputs
    result = parser.parse("red")
    assert result == RED

    result = parser.parse("green")
    assert result == GREEN

    result = parser.parse("blue")
    assert result == BLUE

    # Test invalid input
    try:
        parser.parse("INVALID")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        pass

    # Test levenstein distance matching
    parser = ChoiceOutputParser(options=[RED, GREEN, BLUE], max_distance=2)

    # Test valid inputs
    result = parser.parse("rdd")
    assert result == RED

    result = parser.parse("gren")
    assert result == GREEN

    result = parser.parse("blu")
    assert result == BLUE

    # Test invalid input
    try:
        parser.parse("RED")  # case sensitive
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        pass

    try:
        parser.parse("INVALID")
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        pass
