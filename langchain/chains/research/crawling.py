"""Module contains code for crawling a blob (e.g., HTML file) for links.

The main idea behind the crawling module is to identify additional links
that are worth exploring to find more documents that may be relevant for being
able to answer the question correctly.
"""
import urllib.parse
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup, PageElement

from langchain.base_language import BaseLanguageModel
from langchain.chains.classification.multiselection import MultiSelectChain
from langchain.chains.research.typedefs import BlobCrawler
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.parsers.html.markdownify import MarkdownifyHTMLParser


def _get_ahref_snippets(html: str, num_chars: int = 0) -> Dict[str, Any]:
    """Get a list of <a> tags as snippets from the given html.

    Args:
        html: the html to get snippets from.
        num_chars: the number of characters to get around the <a> tags.

    Returns:
        a list of snippets.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip()
    snippets = []

    for idx, a_tag in enumerate(soup.find_all("a")):
        before_text = _get_surrounding_text(a_tag, num_chars, is_before=True)
        after_text = _get_surrounding_text(a_tag, num_chars, is_before=False)
        snippet = {
            "id": idx,
            "before": before_text.strip().replace("\n", " "),
            "link": a_tag.get("href").replace("\n", " ").strip(),
            "content": a_tag.text.replace("\n", " ").strip(),
            "after": after_text.strip().replace("\n", " "),
        }
        snippets.append(snippet)

    return {
        "snippets": snippets,
        "title": title,
    }


def _extract_records(blob: Blob) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:
    """Extract records from a blob."""
    if blob.mimetype == "text/html":
        info = _get_ahref_snippets(blob.as_string(), num_chars=100)
        return (
            [
                {
                    "content": d["content"],
                    "link": d["link"],
                    "before": d["before"],
                    "after": d["after"],
                }
                for d in info["snippets"]
            ],
            ("link", "content", "before", "after"),
        )
    else:
        raise ValueError(
            "Can only extract records from HTML/JSON blobs. Got {blob.mimetype}"
        )


class ChainCrawler(BlobCrawler):
    def __init__(self, chain: MultiSelectChain, parser: BaseBlobParser) -> None:
        """Crawl the blob using an LLM."""
        self.chain = chain
        self.parser = parser

    def crawl(self, blob: Blob, question: str) -> List[str]:
        """Explore the blob and suggest additional content to explore."""
        if not blob.source:
            raise NotImplementedError()
        records, columns = _extract_records(blob)

        result = self.chain(
            inputs={"question": question, "choices": records, "columns": columns},
        )

        selected_records = result["selected"]

        urls = [
            # TODO(): handle absolute links
            urllib.parse.urljoin(blob.source, record["link"])
            for record in selected_records
            if "mailto:" not in record["link"]
        ]
        return urls

    @classmethod
    def from_default(
        cls,
        llm: BaseLanguageModel,
        blob_parser: BaseBlobParser = MarkdownifyHTMLParser(),
    ) -> "ChainCrawler":
        """Create a crawler from the default LLM."""
        chain = MultiSelectChain.from_default(llm)
        return cls(chain=chain, parser=blob_parser)


def _get_surrounding_text(tag: PageElement, n: int, *, is_before: bool = True) -> str:
    """Get surrounding text the given tag in the given direction.

    Args:
        tag: the tag to get surrounding text for.
        n: number of characters to get
        is_before: Whether to get text before or after the tag.

    Returns:
        the surrounding text in the given direction.
    """
    text = ""
    current = tag.previous_element if is_before else tag.next_element

    while current and len(text) < n:
        current_text = str(current.text).strip()
        current_text = (
            current_text
            if len(current_text) + len(text) <= n
            else current_text[: n - len(text)]
        )

        if is_before:
            text = current_text + " " + text
        else:
            text = text + " " + current_text

        current = current.previous_element if is_before else current.next_element

    return text
