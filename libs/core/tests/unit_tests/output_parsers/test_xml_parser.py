"""Test XMLOutputParser"""
import defusedxml
import pytest

from langchain_core.exceptions import OutputParserException
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

    with pytest.raises(OutputParserException) as e:
        xml_parser.parse(result)
    assert "Failed to parse" in str(e)


def tests_billion_laughs_attack() -> None:
    payload = """<?xml version="1.0"?>
    <!DOCTYPE lolz [
     <!ENTITY lol "lol">
     <!ELEMENT lolz (#PCDATA)>
     <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
     <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
     <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
     <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
     <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">
     <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">
     <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">
     <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">
     <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">
    ]>
    <lolz>&lol9;</lolz>"""
    parser = XMLOutputParser()
    with pytest.raises(defusedxml.EntitiesForbidden):
        parser.parse(payload)
