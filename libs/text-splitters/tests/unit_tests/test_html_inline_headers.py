"""Heading metadata must survive inline markup in documentation pages."""

import pytest
from langchain_core.documents import Document

from langchain_text_splitters import HTMLHeaderTextSplitter


@pytest.mark.parametrize("return_each_element", [False, True])
@pytest.mark.parametrize(
    ("heading", "title"),
    [
        ('<a href="#usage">Usage notes</a>', "Usage notes"),
        ("Use <code>pip</code> to install", "Use pip to install"),
        ('<!--lit-node 1--><a href="#usage">Usage <em>notes</em></a>', "Usage notes"),
    ],
)
def test_inline_heading_metadata(
    *, heading: str, title: str, return_each_element: bool
) -> None:
    splitter = HTMLHeaderTextSplitter(
        [("h1", "title"), ("h2", "section")],
        return_each_element=return_each_element,
    )
    html = f"<h1>Guide</h1><h2>{heading}</h2><p>Section content.</p>"
    assert splitter.split_text(html) == [
        Document(page_content="Guide", metadata={"title": "Guide"}),
        Document(page_content=title, metadata={"title": "Guide", "section": title}),
        Document(
            page_content="Section content.",
            metadata={"title": "Guide", "section": title},
        ),
    ]


@pytest.mark.parametrize("return_each_element", [False, True])
def test_inline_headers_keep_section_boundaries(*, return_each_element: bool) -> None:
    splitter = HTMLHeaderTextSplitter(
        [("h1", "title"), ("h2", "section"), ("h3", "subsection")],
        return_each_element=return_each_element,
    )
    documents = splitter.split_text(
        '<h1><code>pkg</code> guide</h1><h2><a href="#setup">Setup</a></h2>'
        "<p>Install package.</p><h3><strong>Linux</strong></h3><p>Run setup.</p>"
        '<h2><a href="#usage">Usage</a></h2><p>Run package.</p>'
    )
    assert documents == [
        Document(page_content="pkg guide", metadata={"title": "pkg guide"}),
        Document(
            page_content="Setup", metadata={"title": "pkg guide", "section": "Setup"}
        ),
        Document(
            page_content="Install package.",
            metadata={"title": "pkg guide", "section": "Setup"},
        ),
        Document(
            page_content="Linux",
            metadata={"title": "pkg guide", "section": "Setup", "subsection": "Linux"},
        ),
        Document(
            page_content="Run setup.",
            metadata={"title": "pkg guide", "section": "Setup", "subsection": "Linux"},
        ),
        Document(
            page_content="Usage", metadata={"title": "pkg guide", "section": "Usage"}
        ),
        Document(
            page_content="Run package.",
            metadata={"title": "pkg guide", "section": "Usage"},
        ),
    ]
