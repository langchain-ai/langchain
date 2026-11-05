# Copyright (c) 2025 LangChain Contributors
# Licensed under the MIT License

"""Regression tests for nested heading text in HTMLHeaderTextSplitter."""

from langchain_text_splitters import HTMLHeaderTextSplitter


class TestHeaderNestedText:
    """Headers with inline markup (<code>, <a>, <em>) must keep all their text.

    The old code extracted text with recursive=False, which only finds direct
    string children. For `<h1><code>pkg</code> guide</h1>`, the "pkg" text
    inside <code> was not a direct string child and was lost.
    """

    def test_code_inside_h1(self) -> None:
        html = "<h1><code>pkg</code> guide</h1><p>body text</p>"
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "title")])
        docs = splitter.split_text(html)
        titles = [d.metadata["title"] for d in docs if "title" in d.metadata]
        assert "pkg guide" in titles

    def test_link_inside_h2(self) -> None:
        html = '<h2><a href="#usage">Usage notes</a></h2><p>body</p>'
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "section")])
        docs = splitter.split_text(html)
        sections = [d.metadata["section"] for d in docs if "section" in d.metadata]
        assert "Usage notes" in sections

    def test_plain_heading_control(self) -> None:
        """Plain headings without nested markup still work."""
        html = "<h1>Introduction</h1><p>Welcome.</p>"
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "title")])
        docs = splitter.split_text(html)
        titles = [d.metadata["title"] for d in docs if "title" in d.metadata]
        assert "Introduction" in titles
