from __future__ import annotations

import copy
import pathlib
from io import BytesIO, StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

import requests
from langchain_core.documents import Document

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class ElementType(TypedDict):
    """Element type as typed dict."""

    url: str
    xpath: str
    content: str
    metadata: Dict[str, str]


class HTMLHeaderTextSplitter:
    """
    Splitting HTML files based on specified headers.
    Requires lxml package.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_element: bool = False,
    ):
        """Create a new HTMLHeaderTextSplitter.

        Args:
            headers_to_split_on: list of tuples of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2)].
            return_each_element: Return each element w/ associated headers.
        """
        # Output element-by-element or aggregated into chunks w/ common headers
        self.return_each_element = return_each_element
        self.headers_to_split_on = sorted(headers_to_split_on)

    def aggregate_elements_to_chunks(
        self, elements: List[ElementType]
    ) -> List[Document]:
        """Combine elements with common metadata into chunks

        Args:
            elements: HTML element content with associated identifying info and metadata
        """
        aggregated_chunks: List[ElementType] = []

        for element in elements:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == element["metadata"]
            ):
                # If the last element in the aggregated list
                # has the same metadata as the current element,
                # append the current content to the last element's content
                aggregated_chunks[-1]["content"] += "  \n" + element["content"]
            else:
                # Otherwise, append the current element to the aggregated list
                aggregated_chunks.append(element)

        return [
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
            for chunk in aggregated_chunks
        ]

    def split_text_from_url(self, url: str, **kwargs: Any) -> List[Document]:
        """Split HTML from web URL

        Args:
            url: web URL
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the fetch url content request.
        """
        r = requests.get(url, **kwargs)
        return self.split_text_from_file(BytesIO(r.content))

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string

        Args:
            text: HTML text
        """
        return self.split_text_from_file(StringIO(text))

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML file

        Args:
            file: HTML file
        """
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        # Explicitly encoding in utf-8 allows non-English
        # html files to be processed without garbled characters
        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.parse(file, parser)

        # document transformation for "structure-aware" chunking is handled with xsl.
        # see comments in html_chunks_with_headers.xslt for more detailed information.
        xslt_path = pathlib.Path(__file__).parent / "xsl/html_chunks_with_headers.xslt"
        xslt_tree = etree.parse(xslt_path)
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        result_dom = etree.fromstring(str(result))

        # create filter and mapping for header metadata
        header_filter = [header[0] for header in self.headers_to_split_on]
        header_mapping = dict(self.headers_to_split_on)

        # map xhtml namespace prefix
        ns_map = {"h": "http://www.w3.org/1999/xhtml"}

        # build list of elements from DOM
        elements = []
        for element in result_dom.findall("*//*", ns_map):
            if element.findall("*[@class='headers']") or element.findall(
                "*[@class='chunk']"
            ):
                elements.append(
                    ElementType(
                        url=file,
                        xpath="".join(
                            [
                                node.text or ""
                                for node in element.findall("*[@class='xpath']", ns_map)
                            ]
                        ),
                        content="".join(
                            [
                                node.text or ""
                                for node in element.findall("*[@class='chunk']", ns_map)
                            ]
                        ),
                        metadata={
                            # Add text of specified headers to metadata using header
                            # mapping.
                            header_mapping[node.tag]: node.text or ""
                            for node in filter(
                                lambda x: x.tag in header_filter,
                                element.findall("*[@class='headers']/*", ns_map),
                            )
                        },
                    )
                )

        if not self.return_each_element:
            return self.aggregate_elements_to_chunks(elements)
        else:
            return [
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
                for chunk in elements
            ]


class HTMLSectionSplitter:
    """
    Splitting HTML files based on specified tag and font sizes.
    Requires lxml package.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        xslt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new HTMLSectionSplitter.

        Args:
            headers_to_split_on: list of tuples of headers we want to track mapped to
                (arbitrary) keys for metadata. Allowed header values: h1, h2, h3, h4,
                h5, h6 e.g. [("h1", "Header 1"), ("h2", "Header 2"].
            xslt_path: path to xslt file for document transformation.
            Uses a default if not passed.
            Needed for html contents that using different format and layouts.
        """
        self.headers_to_split_on = dict(headers_to_split_on)

        if xslt_path is None:
            self.xslt_path = (
                pathlib.Path(__file__).parent / "xsl/converting_to_header.xslt"
            ).absolute()
        else:
            self.xslt_path = pathlib.Path(xslt_path).absolute()
        self.kwargs = kwargs

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        results = self.create_documents(texts, metadatas=metadatas)

        text_splitter = RecursiveCharacterTextSplitter(**self.kwargs)

        return text_splitter.split_documents(results)

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string

        Args:
            text: HTML text
        """
        return self.split_text_from_file(StringIO(text))

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])

                for key in chunk.metadata.keys():
                    if chunk.metadata[key] == "#TITLE#":
                        chunk.metadata[key] = metadata["Title"]
                metadata = {**metadata, **chunk.metadata}
                new_doc = Document(page_content=chunk.page_content, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_html_by_headers(self, html_doc: str) -> List[Dict[str, Optional[str]]]:
        try:
            from bs4 import BeautifulSoup, PageElement  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Unable to import BeautifulSoup/PageElement, \
                    please install with `pip install \
                    bs4`."
            ) from e

        soup = BeautifulSoup(html_doc, "html.parser")
        headers = list(self.headers_to_split_on.keys())
        sections: list[dict[str, str | None]] = []

        headers = soup.find_all(["body"] + headers)

        for i, header in enumerate(headers):
            header_element: PageElement = header
            if i == 0:
                current_header = "#TITLE#"
                current_header_tag = "h1"
                section_content: List = []
            else:
                current_header = header_element.text.strip()
                current_header_tag = header_element.name
                section_content = []
            for element in header_element.next_elements:
                if i + 1 < len(headers) and element == headers[i + 1]:
                    break
                if isinstance(element, str):
                    section_content.append(element)
            content = " ".join(section_content).strip()

            if content != "":
                sections.append(
                    {
                        "header": current_header,
                        "content": content,
                        "tag_name": current_header_tag,
                    }
                )

        return sections

    def convert_possible_tags_to_header(self, html_content: str) -> str:
        if self.xslt_path is None:
            return html_content

        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(
                "Unable to import lxml, please install with `pip install lxml`."
            ) from e
        # use lxml library to parse html document and return xml ElementTree
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(html_content), parser)

        xslt_tree = etree.parse(self.xslt_path)
        transform = etree.XSLT(xslt_tree)
        result = transform(tree)
        return str(result)

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML file

        Args:
            file: HTML file
        """
        file_content = file.getvalue()
        file_content = self.convert_possible_tags_to_header(file_content)
        sections = self.split_html_by_headers(file_content)

        return [
            Document(
                cast(str, section["content"]),
                metadata={
                    self.headers_to_split_on[str(section["tag_name"])]: section[
                        "header"
                    ]
                },
            )
            for section in sections
        ]
