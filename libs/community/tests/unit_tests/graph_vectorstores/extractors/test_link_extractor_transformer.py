from typing import Set

from langchain_core.documents import Document

from langchain_community.graph_vectorstores.extractors import (
    LinkExtractor,
    LinkExtractorTransformer,
)
from langchain_community.graph_vectorstores.links import Link, get_links

TEXT1 = "Text1"
TEXT2 = "Text2"


class FakeKeywordExtractor(LinkExtractor[Document]):
    def extract_one(self, input: Document) -> Set[Link]:
        kws: Set[str] = set()
        if input.page_content == TEXT1:
            kws = {"a", "b"}
        elif input.page_content == TEXT2:
            kws = {"b", "c"}

        return {Link.bidir(kind="fakekw", tag=kw) for kw in kws}


class FakeHyperlinkExtractor(LinkExtractor[Document]):
    def extract_one(self, input: Document) -> Set[Link]:
        if input.page_content == TEXT1:
            return {
                Link.incoming(kind="fakehref", tag="http://text1"),
                Link.outgoing(kind="fakehref", tag="http://text2"),
                Link.outgoing(kind="fakehref", tag="http://text3"),
            }
        elif input.page_content == TEXT2:
            return {
                Link.incoming(kind="fakehref", tag="http://text2"),
                Link.outgoing(kind="fakehref", tag="http://text3"),
            }
        else:
            raise ValueError(
                f"Unsupported input for FakeHyperlinkExtractor: '{input.page_content}'"
            )


def test_one_extractor() -> None:
    transformer = LinkExtractorTransformer(
        [
            FakeKeywordExtractor(),
        ]
    )
    doc1 = Document(TEXT1)
    doc2 = Document(TEXT2)
    results = transformer.transform_documents([doc1, doc2])

    assert set(get_links(results[0])) == {
        Link.bidir(kind="fakekw", tag="a"),
        Link.bidir(kind="fakekw", tag="b"),
    }

    assert set(get_links(results[1])) == {
        Link.bidir(kind="fakekw", tag="b"),
        Link.bidir(kind="fakekw", tag="c"),
    }


def test_multiple_extractors() -> None:
    transformer = LinkExtractorTransformer(
        [
            FakeKeywordExtractor(),
            FakeHyperlinkExtractor(),
        ]
    )

    doc1 = Document(TEXT1)
    doc2 = Document(TEXT2)

    results = transformer.transform_documents([doc1, doc2])

    assert set(get_links(results[0])) == {
        Link.bidir(kind="fakekw", tag="a"),
        Link.bidir(kind="fakekw", tag="b"),
        Link.incoming(kind="fakehref", tag="http://text1"),
        Link.outgoing(kind="fakehref", tag="http://text2"),
        Link.outgoing(kind="fakehref", tag="http://text3"),
    }

    assert set(get_links(results[1])) == {
        Link.bidir(kind="fakekw", tag="b"),
        Link.bidir(kind="fakekw", tag="c"),
        Link.incoming(kind="fakehref", tag="http://text2"),
        Link.outgoing(kind="fakehref", tag="http://text3"),
    }
