from __future__ import annotations

import copy
import pathlib
from io import BytesIO, StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from langchain.docstore.document import Document

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
        """Split HTML file using BeautifulSoup.
        Args:
            file: HTML file path or file-like object.
        Returns:
            List of Document objects with page_content and metadata.
        """
        # Read the HTML content from the file or file-like object
        if isinstance(file, str):
            with open(file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            # Assuming file is a file-like object
            html_content = file.read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract the header tags and their corresponding metadata keys
        headers_to_split_on = [tag[0] for tag in self.headers_to_split_on]
        header_mapping = dict(self.headers_to_split_on)

        documents = []

        # Find the body of the document
        body = soup.body if soup.body else soup

        # Find all header tags in the order they appear
        all_headers = body.find_all(headers_to_split_on)

        # If there's content before the first header, collect it
        first_header = all_headers[0] if all_headers else None
        if first_header:
            pre_header_content = ''
            for elem in first_header.find_all_previous():
                if isinstance(elem, bs4.Tag):
                    text = elem.get_text(separator=' ', strip=True)
                    if text:
                        pre_header_content = text + ' ' + pre_header_content
            if pre_header_content.strip():
                documents.append(Document(
                    page_content=pre_header_content.strip(),
                    metadata={}  # No metadata since there's no header
                ))
        else:
            # If no headers are found, return the whole content
            full_text = body.get_text(separator=' ', strip=True)
            if full_text.strip():
                documents.append(Document(
                    page_content=full_text.strip(),
                    metadata={}
                ))
            return documents

        # Process each header and its associated content
        for header in all_headers:
            current_metadata = {}
            header_name = header.name
            header_text = header.get_text(separator=' ', strip=True)
            current_metadata[header_mapping[header_name]] = header_text

            # Collect all sibling elements until the next header of the same or higher level
            content_elements = []
            for sibling in header.find_next_siblings():
                if sibling.name in headers_to_split_on:
                    # Stop at the next header
                    break
                if isinstance(sibling, bs4.Tag):
                    content_elements.append(sibling)

            # Get the text content of the collected elements
            current_content = ''
            for elem in content_elements:
                text = elem.get_text(separator=' ', strip=True)
                if text:
                    current_content += text + ' '

            # Create a Document if there is content
            if current_content.strip():
                documents.append(Document(
                    page_content=current_content.strip(),
                    metadata=current_metadata.copy()
                ))
            else:
                # If there's no content, but we have metadata, still create a Document
                documents.append(Document(
                    page_content='',
                    metadata=current_metadata.copy()
                ))

        return documents


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
