"""Test XMLOutputParser."""

import importlib
import time
from collections.abc import AsyncIterator, Iterable

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


async def _test_parser(parser: XMLOutputParser, content: str) -> None:
    """Test parser."""
    assert parser.parse(content) == DEF_RESULT_EXPECTED
    assert await parser.aparse(content) == DEF_RESULT_EXPECTED

    assert list(parser.transform(iter(content))) == [
        {"foo": [{"bar": [{"baz": None}]}]},
        {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
        {"foo": [{"baz": "tag"}]},
    ]

    chunks = [chunk async for chunk in parser.atransform(_as_iter(content))]

    assert list(chunks) == [
        {"foo": [{"bar": [{"baz": None}]}]},
        {"foo": [{"bar": [{"baz": "slim.shady"}]}]},
        {"foo": [{"baz": "tag"}]},
    ]


ROOT_LEVEL_ONLY = """<?xml version="1.0" encoding="UTF-8"?>
<body>Text of the body.</body>
"""

ROOT_LEVEL_ONLY_EXPECTED = {"body": "Text of the body."}


async def _as_iter(iterable: Iterable[str]) -> AsyncIterator[str]:
    for item in iterable:
        yield item


async def test_root_only_xml_output_parser() -> None:
    """Test XMLOutputParser when xml only contains the root level tag."""
    xml_parser = XMLOutputParser(parser="xml")
    assert xml_parser.parse(ROOT_LEVEL_ONLY) == {"body": "Text of the body."}
    assert await xml_parser.aparse(ROOT_LEVEL_ONLY) == {"body": "Text of the body."}
    assert list(xml_parser.transform(iter(ROOT_LEVEL_ONLY))) == [
        {"body": "Text of the body."}
    ]
    chunks = [chunk async for chunk in xml_parser.atransform(_as_iter(ROOT_LEVEL_ONLY))]
    assert chunks == [{"body": "Text of the body."}]


@pytest.mark.parametrize(
    "content",
    [
        DATA,  # has no xml header
        WITH_XML_HEADER,
        IN_XML_TAGS_WITH_XML_HEADER,
        IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK,
    ],
)
async def test_xml_output_parser(content: str) -> None:
    """Test XMLOutputParser."""
    xml_parser = XMLOutputParser(parser="xml")
    await _test_parser(xml_parser, content)


@pytest.mark.skipif(
    importlib.util.find_spec("defusedxml") is None,
    reason="defusedxml is not installed",
)
@pytest.mark.parametrize(
    "content",
    [
        DATA,  # has no xml header
        WITH_XML_HEADER,
        IN_XML_TAGS_WITH_XML_HEADER,
        IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK,
    ],
)
async def test_xml_output_parser_defused(content: str) -> None:
    """Test XMLOutputParser."""
    xml_parser = XMLOutputParser(parser="defusedxml")
    await _test_parser(xml_parser, content)


ELEMENT_WITH_ENCODING_ATTRIBUTE = """<root encoding="utf-8">
<item>x</item>
</root>"""

ELEMENT_WITH_ENCODING_ATTRIBUTE_EXPECTED = {"root": [{"item": "x"}]}


@pytest.mark.parametrize(
    "backend",
    [
        "xml",
        pytest.param(
            "defusedxml",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("defusedxml") is None,
                reason="defusedxml is not installed",
            ),
        ),
    ],
)
async def test_xml_output_parser_element_with_encoding_attribute(
    backend: str,
) -> None:
    """An element whose opening tag has an `encoding` attribute must be kept.

    Only an XML declaration (`<?xml ... encoding="..."?>`) should be stripped
    before parsing, not an ordinary element that happens to carry an
    `encoding` attribute followed by a newline.
    """
    xml_parser = XMLOutputParser(parser=backend)
    expected = ELEMENT_WITH_ENCODING_ATTRIBUTE_EXPECTED
    assert xml_parser.parse(ELEMENT_WITH_ENCODING_ATTRIBUTE) == expected
    assert await xml_parser.aparse(ELEMENT_WITH_ENCODING_ATTRIBUTE) == expected
    assert list(xml_parser.transform(iter(ELEMENT_WITH_ENCODING_ATTRIBUTE))) == [
        expected
    ]


def test_xml_output_parser_encoding_declaration_is_stripped() -> None:
    """The XML declaration is still removed when it precedes the document."""
    xml_parser = XMLOutputParser(parser="xml")
    content = '<?xml version="1.0" encoding="UTF-8"?>\n<root>\n<item>x</item>\n</root>'
    assert xml_parser.parse(content) == ELEMENT_WITH_ENCODING_ATTRIBUTE_EXPECTED
    match = xml_parser.encoding_matcher.search(content)
    assert match is not None
    assert match.group(2) == "<root>\n<item>x</item>\n</root>"


def test_xml_output_parser_many_open_brackets_is_linear() -> None:
    """Searching for the encoding declaration must not be quadratic.

    Model output is untrusted. A string of `<` characters with no `>` used to
    make the declaration regex rescan to the end of the text from every `<`,
    which took seconds for a few tens of thousands of characters.
    """
    xml_parser = XMLOutputParser(parser="xml")
    content = "<" * 300_000

    start = time.perf_counter()
    with pytest.raises(OutputParserException):
        xml_parser.parse(content)
    elapsed = time.perf_counter() - start

    # Quadratic matching takes well over 30 seconds on this input; a linear
    # scan takes milliseconds. The bound is loose to stay robust on slow CI.
    assert elapsed < 5


@pytest.mark.parametrize("result", ["foo></foo>", "<foo></foo", "foo></foo", "foofoo"])
def test_xml_output_parser_fail(result: str) -> None:
    """Test XMLOutputParser where complete output is not in XML format."""
    xml_parser = XMLOutputParser(parser="xml")

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
    # Testing with standard XML parser since it's safe to use in
    # newer versions of Python
    parser = XMLOutputParser(parser="xml")
    with pytest.raises(OutputParserException):
        parser.parse(MALICIOUS_XML)

    with pytest.raises(OutputParserException):
        await parser.aparse(MALICIOUS_XML)
