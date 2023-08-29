"""Test XMLOutputParser"""
import pytest

from langchain.output_parsers.xml import XMLOutputParser

DEF_RESULT_ENCODING = """<?xml version="1.0" encoding="UTF-8"?>
<foo>
    <bar>
        <baz></baz>
        <baz></baz>
    </bar>  
</foo>"""

DEF_RESULT_EXPECTED = """<foo>
    <bar>
        <baz></baz>
        <baz></baz>
    </bar>  
</foo>"""


@pytest.mark.parametrize("result", [DEF_RESULT_ENCODING, DEF_RESULT_EXPECTED])
def test_xml_output_parser(result: str) -> None:
    """Test XMLOutputParser."""

    xml_parser = XMLOutputParser()

    result = xml_parser.parse(result)
    print("parse_result:", result)
    assert DEF_RESULT_EXPECTED == result


@pytest.mark.parametrize("result", ["foo></foo>", "<foo></foo", "foo></foo", "foofoo"])
def test_xml_output_parser_fail(result: str) -> None:
    """Test XMLOutputParser where complete output is not in XML format."""

    xml_parser = XMLOutputParser()

    try:
        xml_parser.parse(result)
    except ValueError as e:
        print("parse_result:", e)
        assert "Could not parse output" in str(e)
    else:
        assert False, "Expected ValueError"
