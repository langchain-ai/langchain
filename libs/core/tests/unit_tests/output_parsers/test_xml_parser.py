"""Test XMLOutputParser"""
from typing import AsyncIterator
from xml.etree.ElementTree import ParseError

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
async def test_xml_output_parser(result: str) -> None:
    """Test XMLOutputParser."""

    xml_parser = XMLOutputParser()
    assert DEF_RESULT_EXPECTED == xml_parser.parse(result)
    assert DEF_RESULT_EXPECTED == (await xml_parser.aparse(result))
    assert list(xml_parser.transform(iter(result))) == [
        {"foo": [{"bar": [{"baz": None}]}]},
        {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
        {"foo": [{"baz": "tag"}]},
    ]

    async def _as_iter(string: str) -> AsyncIterator[str]:
        for c in string:
            yield c

    chunks = [chunk async for chunk in xml_parser.atransform(_as_iter(result))]
    assert chunks == [
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


MALICIOUS_XML = """<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ELEMENT lolz (#PCDATA)>
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


async def tests_billion_laughs_attack() -> None:
    parser = XMLOutputParser()
    with pytest.raises(OutputParserException):
        parser.parse(MALICIOUS_XML)

    with pytest.raises(OutputParserException):
        await parser.aparse(MALICIOUS_XML)

    with pytest.raises(ParseError):
        # Right now raises undefined entity error
        assert list(parser.transform(iter(MALICIOUS_XML))) == [
            {"foo": [{"bar": [{"baz": None}]}]}
        ]

    async def _as_iter(string: str) -> AsyncIterator[str]:
        for c in string:
            yield c

    with pytest.raises(ParseError):
        chunks = [chunk async for chunk in parser.atransform(_as_iter(MALICIOUS_XML))]
        assert chunks == [{"foo": [{"bar": [{"baz": None}]}]}]
