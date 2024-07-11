import pytest

from langchain_community.graph_vectorstores import GraphStoreLink
from langchain_community.graph_vectorstores.extractors import (
    HtmlInput,
    HtmlLinkExtractor,
)

PAGE_1 = """
<html>
<body>
Hello.
<a href="relative">Relative</a>
<a href="/relative-base">Relative base.</a>
<a href="http://cnn.com">Aboslute</a>
<a href="//same.foo">Test</a>
</body>
</html>
"""

PAGE_2 = """
<html>
<body>
Hello.
<a href="/bar/#fragment">Relative</a>
</html>
"""


@pytest.mark.requires("bs4")
def test_one_from_str() -> None:
    extractor = HtmlLinkExtractor()

    results = extractor.extract_one(HtmlInput(PAGE_1, base_url="https://foo.com/bar/"))
    assert results == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/bar/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/relative"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/relative-base"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://cnn.com"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://same.foo"),
    }

    results = extractor.extract_one(HtmlInput(PAGE_1, base_url="http://foo.com/bar/"))
    assert results == {
        GraphStoreLink.incoming(kind="hyperlink", tag="http://foo.com/bar/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://foo.com/bar/relative"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://foo.com/relative-base"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://cnn.com"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://same.foo"),
    }


@pytest.mark.requires("bs4")
def test_one_from_beautiful_soup() -> None:
    from bs4 import BeautifulSoup

    extractor = HtmlLinkExtractor()
    soup = BeautifulSoup(PAGE_1, "html.parser")
    results = extractor.extract_one(HtmlInput(soup, base_url="https://foo.com/bar/"))
    assert results == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/bar/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/relative"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/relative-base"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://cnn.com"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://same.foo"),
    }


@pytest.mark.requires("bs4")
def test_drop_fragments() -> None:
    extractor = HtmlLinkExtractor(drop_fragments=True)
    results = extractor.extract_one(
        HtmlInput(PAGE_2, base_url="https://foo.com/baz/#fragment")
    )

    assert results == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/baz/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/"),
    }


@pytest.mark.requires("bs4")
def test_include_fragments() -> None:
    extractor = HtmlLinkExtractor(drop_fragments=False)
    results = extractor.extract_one(
        HtmlInput(PAGE_2, base_url="https://foo.com/baz/#fragment")
    )

    assert results == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/baz/#fragment"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/#fragment"),
    }


@pytest.mark.requires("bs4")
def test_batch_from_str() -> None:
    extractor = HtmlLinkExtractor()
    results = list(
        extractor.extract_many(
            [
                HtmlInput(PAGE_1, base_url="https://foo.com/bar/"),
                HtmlInput(PAGE_2, base_url="https://foo.com/baz/"),
            ]
        )
    )

    assert results[0] == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/bar/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/relative"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/relative-base"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="http://cnn.com"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://same.foo"),
    }
    assert results[1] == {
        GraphStoreLink.incoming(kind="hyperlink", tag="https://foo.com/baz/"),
        GraphStoreLink.outgoing(kind="hyperlink", tag="https://foo.com/bar/"),
    }
