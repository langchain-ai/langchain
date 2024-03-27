"""Test XMLOutputParser"""
from typing import AsyncIterator, Iterable

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.xml import XMLOutputParser

DATA = """
 <foo>
    <bar>
        <baz></baz>
        <baz>slim.shady</baz>
    </bar>
    <baz>tag</baz>
</foo>"""

WITH_XML_HEADER = f"""<?xml version="1.0" encoding="UTF-8"?>
{DATA}"""


IN_XML_TAGS_WITH_XML_HEADER = f"""
```xml
{WITH_XML_HEADER}
```
"""

IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK = f"""
Some random text
```xml
{WITH_XML_HEADER}
```
More random text
"""


DEF_RESULT_EXPECTED = {
    "foo": [
        {"bar": [{"baz": None}, {"baz": "slim.shady"}]},
        {"baz": "tag"},
    ],
}


@pytest.mark.parametrize(
    "result",
    [
        DATA,  # has no xml header
        WITH_XML_HEADER,
        IN_XML_TAGS_WITH_XML_HEADER,
        IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK,
    ],
)
async def test_xml_output_parser(result: str) -> None:
    """Test XMLOutputParser."""

    xml_parser = XMLOutputParser()

    xml_result = xml_parser.parse(result)
    assert DEF_RESULT_EXPECTED == xml_result

    assert list(xml_parser.transform(iter(result))) == [
        {"foo": [{"bar": [{"baz": None}]}]},
        {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
        {"foo": [{"baz": "tag"}]},
    ]

    async def _as_iter(iterable: Iterable[str]) -> AsyncIterator[str]:
        for item in iterable:
            yield item

    chunks = [chunk async for chunk in xml_parser.atransform(_as_iter(result))]

    assert list(chunks) == [
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
