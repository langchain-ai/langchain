from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.item_parsed_list import ItemParsedListOutputParser


def test_parsed_list_output_parser_parse() -> None:
    boolean_output_parser = BooleanOutputParser()
    item_parsed_list_output_parser = ItemParsedListOutputParser(
        item_parser=boolean_output_parser,
        item_name="boolean value",
    )

    # Test valid input
    result = item_parsed_list_output_parser.parse(
        f"{boolean_output_parser.true_val}\n{boolean_output_parser.false_val}"
    )
    assert result == [True, False]

    # Test valid input
    result = item_parsed_list_output_parser.parse(
        f"{boolean_output_parser.false_val}\n{boolean_output_parser.true_val}\n{boolean_output_parser.false_val}"
    )
    assert result == [False, True, False]

    # Test invalid input
    try:
        item_parsed_list_output_parser.parse("INVALID")
        assert False, "Should have raised ValueError"  # from BooleanOutputParser
    except ValueError:
        pass

    # Test custom custom separator
    item_parsed_list_output_parser.separator = ", "

    # Test valid input
    result = item_parsed_list_output_parser.parse(
        f"{boolean_output_parser.true_val}, {boolean_output_parser.false_val}"
    )
    assert result == [True, False]

    # Test valid input
    result = item_parsed_list_output_parser.parse(
        f"{boolean_output_parser.false_val}, {boolean_output_parser.true_val}, {boolean_output_parser.false_val}"
    )
    assert result == [False, True, False]

    # Test invalid input
    try:
        item_parsed_list_output_parser.parse("INVALID")
        assert False, "Should have raised ValueError"  # from BooleanOutputParser
    except ValueError:
        pass
