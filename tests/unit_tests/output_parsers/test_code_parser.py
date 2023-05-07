import inspect
from textwrap import dedent
from langchain.output_parsers.code_block import CodeOutputParser
from langchain.schema import OutputParserException


def test_boolean_output_parser_parse() -> None:
    parser = CodeOutputParser()

    # Test valid input
    source = dedent(
        inspect.getsource(CodeOutputParser.get_format_instructions)
    ).strip()  # we'll test on our own code!

    response = dedent(
        """
        Sure, here's one possible implementation of test_boolean_output_parser_parse:
        
        ```python
        {code}
        ```
        """
    ).format(code=source)
    parsed_code = parser.parse(response)
    assert parsed_code == source, "Parsed code should match input code"

    # Test invalid input
    try:
        response = dedent(
            """
            Sure, here's one possible implementation of test_boolean_output_parser_parse:
            
            ```python
            {code}
            """
        ).format(code=source[:20])
        parsed_code = parser.parse(response)
        assert False, "Should have raised OutputParserException"
    except OutputParserException:
        pass
