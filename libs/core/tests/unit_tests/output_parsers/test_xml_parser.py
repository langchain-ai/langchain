"""Test XMLOutputParser"""
import pytest

from langchain_core.output_parsers.xml import XMLOutputParser

DEF_RESULT_ENCODING = """<?xml version="1.0" encoding="UTF-8"?>
 <foo>
    <bar>
        <baz></baz>
        <baz>slim.shady</baz>
    </bar>
    <baz>tag</baz>
</foo>"""

DEF_RESULT_EXPECTED = {
    "foo": [
        {"bar": [{"baz": None}, {"baz": "slim.shady"}]},
        {"baz": "tag"},
    ],
}


@pytest.mark.parametrize(
    "result",
    [
        DEF_RESULT_ENCODING,
        DEF_RESULT_ENCODING[DEF_RESULT_ENCODING.find("\n") :],
        f"""
```xml
{DEF_RESULT_ENCODING}
```
""",
        f"""
Some random text
```xml
{DEF_RESULT_ENCODING}
```
More random text
""",
    ],
)
def test_xml_output_parser(result: str) -> None:
    """Test XMLOutputParser."""

    xml_parser = XMLOutputParser()

    xml_result = xml_parser.parse(result)
    assert DEF_RESULT_EXPECTED == xml_result
    assert list(xml_parser.transform(iter(result))) == [
        {"foo": [{"bar": [{"baz": None}]}]},
        {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
        {"foo": [{"baz": "tag"}]},
    ]


@pytest.mark.parametrize("result", ["foo></foo>", "<foo></foo", "foo></foo", "foofoo"])
def test_xml_output_parser_fail(result: str) -> None:
    """Test XMLOutputParser where complete output is not in XML format."""

    xml_parser = XMLOutputParser()

    with pytest.raises(ValueError) as e:
        xml_parser.parse(result)
    assert "Could not parse output" in str(e)
