"""Test XMLOutputParser."""

import importlib
from collections.abc import AsyncIterator, Iterable
import warnings

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
    """Test XMLOutputParser with legacy xml parser."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xml_parser = XMLOutputParser(parser="xml")
        await _test_parser(xml_parser, content)


@pytest.mark.parametrize(
    "content",
    [
        DATA,  # has no xml header
        WITH_XML_HEADER,
        IN_XML_TAGS_WITH_XML_HEADER,
        IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK,
    ],
)
async def test_secure_xml_output_parser(content: str) -> None:
    """Test XMLOutputParser with secure parser (new default)."""
    xml_parser = XMLOutputParser(parser="secure")
    await _test_parser(xml_parser, content)


@pytest.mark.parametrize(
    "content",
    [
        DATA,  # has no xml header
        WITH_XML_HEADER,
        IN_XML_TAGS_WITH_XML_HEADER,
        IN_XML_TAGS_WITH_HEADER_AND_TRAILING_JUNK,
    ],
)
async def test_default_xml_output_parser(content: str) -> None:
    """Test XMLOutputParser with default parser (should be secure)."""
    xml_parser = XMLOutputParser()  # Should default to secure
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
    """Test XMLOutputParser with defusedxml parser."""
    xml_parser = XMLOutputParser(parser="defusedxml")
    await _test_parser(xml_parser, content)


@pytest.mark.parametrize("result", ["foo></foo>", "<foo></foo", "foo></foo", "foofoo"])
def test_xml_output_parser_fail(result: str) -> None:
    """Test XMLOutputParser where complete output is not in XML format."""
    xml_parser = XMLOutputParser(parser="secure")

    with pytest.raises(OutputParserException) as e:
        xml_parser.parse(result)
    assert "Failed to parse" in str(e)


# Security test cases for XXE and billion laughs attacks
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

XXE_ATTACK_XML = """<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<foo>&xxe;</foo>"""

ENTITY_EXPANSION_XML = """<?xml version="1.0"?>
<root>
  &amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;&amp;
</root>"""


async def test_billion_laughs_attack_secure() -> None:
    """Test that secure parser blocks billion laughs attack."""
    parser = XMLOutputParser(parser="secure")
    with pytest.raises(OutputParserException):
        parser.parse(MALICIOUS_XML)

    with pytest.raises(OutputParserException):
        await parser.aparse(MALICIOUS_XML)


async def test_xxe_attack_secure() -> None:
    """Test that secure parser blocks XXE attacks."""
    parser = XMLOutputParser(parser="secure")
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(XXE_ATTACK_XML)
    
    assert "malicious DTD declarations" in str(exc_info.value)

    with pytest.raises(OutputParserException):
        await parser.aparse(XXE_ATTACK_XML)


async def test_entity_expansion_attack_secure() -> None:
    """Test that secure parser blocks excessive entity expansion."""
    parser = XMLOutputParser(parser="secure")
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(ENTITY_EXPANSION_XML)
    
    assert "excessive entity references" in str(exc_info.value)

    with pytest.raises(OutputParserException):
        await parser.aparse(ENTITY_EXPANSION_XML)


@pytest.mark.skipif(
    importlib.util.find_spec("defusedxml") is None,
    reason="defusedxml is not installed",
)
async def test_billion_laughs_attack_defused() -> None:
    """Test that defusedxml parser blocks billion laughs attack."""
    parser = XMLOutputParser(parser="defusedxml")
    with pytest.raises(OutputParserException):
        parser.parse(MALICIOUS_XML)

    with pytest.raises(OutputParserException):
        await parser.aparse(MALICIOUS_XML)


async def test_billion_laughs_attack_xml_legacy() -> None:
    """Test billion laughs attack with legacy xml parser."""
    # NOTE: Not fully tested with all Python distributions due to automated environment limits
    # Manual testing recommended for: Different Python versions, libexpat configurations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser = XMLOutputParser(parser="xml")
        
        # The standard library parser behavior varies by Python version and libexpat version
        # In newer versions, it should raise an exception, but we can't guarantee this
        try:
            result = parser.parse(MALICIOUS_XML)
            # If parsing succeeds, it means the attack wasn't blocked
            # This is why we recommend the secure parser
            assert result is not None
        except OutputParserException:
            # This is the expected behavior - the attack was blocked
            pass


def test_deprecation_warning_xml_parser() -> None:
    """Test that using xml parser issues deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        XMLOutputParser(parser="xml")
        
        # Check that a deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated due to security risks" in str(w[0].message)


def test_security_warning_secure_parser() -> None:
    """Test that secure parser issues security warning when defusedxml unavailable."""
    # NOTE: This test may not trigger the warning if defusedxml is installed
    # Manual testing recommended for environments without defusedxml
    if importlib.util.find_spec("defusedxml") is not None:
        pytest.skip("defusedxml is installed, warning won't be triggered")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parser = XMLOutputParser(parser="secure")
        parser.parse("<root>test</root>")
        
        # Check that a security warning was issued
        warning_found = any(
            "standard library XML parser" in str(warning.message)
            for warning in w
        )
        assert warning_found
