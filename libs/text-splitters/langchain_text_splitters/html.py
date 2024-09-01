from __future__ import annotations

import copy
import pathlib
import re
from io import BytesIO, StringIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

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


class HTMLSemanticPreservingSplitter:
    """
    Splits HTML content by headers into generalized chunks, preserving semantic
    structure. If chunks exceed the maximum chunk size,
    uses RecursiveCharacterTextSplitter for further splitting.

    The splitter preserves full HTML elements (e.g., <table>, <ul>) and converts links,
    to Markdown-like links.
    Note that some chunks may exceed the maximum size to maintain semantic integrity.

    Attributes:
        headers_to_split_on (List[Tuple[str, str]]): HTML headers (e.g., "h1", "h2")
        that define content sections.
        max_chunk_size (int): Maximum size for each chunk, with allowance for exceeding
        this limit to preserve semantics.
        chunk_overlap (int): Number of characters to overlap between chunks to ensure
        contextual continuity.
        separators (List[str]): Delimiters used by RecursiveCharacterTextSplitter for
        further splitting.
        elements_to_preserve (List[str]): HTML tags (e.g., <table>, <ul>) to remain
        intact during splitting.
        preserve_links (bool): Converts <a> tags to Markdown links ([text](url)).
        custom_handlers (Dict[str, Callable[[Any], str]]): Optional custom handlers for
        specific HTML tags, allowing tailored extraction or processing.

    Example:

        def custom_iframe_extractor(iframe_tag):
            ```
            Custom handler function to extract the 'src' attribute from an <iframe> tag.
            Converts the iframe to a Markdown-like link: [iframe:<src>](src).

            Args:
                iframe_tag (bs4.element.Tag): The <iframe> tag to be processed.

            Returns:
                str: A formatted string representing the iframe in Markdown-like format.
            ```
            iframe_src = iframe_tag.get('src', '')
            return f"[iframe:{iframe_src}]({iframe_src})"

        text_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
            max_chunk_size=500,
            preserve_links=True,
            custom_handlers={"iframe": custom_iframe_extractor}
        )
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        max_chunk_size: int = 1000,
        chunk_overlap: int = 0,
        separators: Optional[List[str]] = None,
        elements_to_preserve: List[str] = [],
        preserve_links: bool = False,
        custom_handlers: Optional[Dict[str, Callable[[Any], str]]] = None,
    ):
        """
        Initializes the HTMLSemanticPreservingSplitter with the provided configuration
        to handle complex HTML splitting scenarios.

        Args:
            headers_to_split_on (List[Tuple[str, str]]): Headers to guide content
            splitting.
            max_chunk_size (int): Upper limit for each content chunk.
            chunk_overlap (int): Amount of overlap between chunks to maintain context.
            separators (Optional[List[str]]): Delimiters RecursiveCharacterTextSplitter.
            elements_to_preserve (List[str]): HTML tags to preserve during splitting.
            preserve_links (bool): Converts links to Markdown format.
            custom_handlers (Optional[Dict[str, Callable[[Any], str]]]): Handlers for
                custom processing of specific HTML tags,
                where the key is the tag name and the value is the processing function.
        """
        try:
            from bs4 import BeautifulSoup, Tag

            self._BeautifulSoup = BeautifulSoup
            self._Tag = Tag
        except ImportError:
            raise ImportError(
                "Could not import BeautifulSoup. \
                              Please install it with 'pip install bs4'."
            )

        self._headers_to_split_on = sorted(headers_to_split_on)
        self._max_chunk_size = max_chunk_size
        self._elements_to_preserve = elements_to_preserve
        self._preserve_links = preserve_links
        self._custom_handlers = custom_handlers or {}
        if separators:
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self._recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
            )

    def split_text(self, text: str) -> List[Document]:
        """
        Splits the provided HTML text into smaller chunks based on the configuration.

        Args:
            text (str): The HTML content to be split.

        Returns:
            List[Document]: A list of Document objects containing the split content.
        """
        soup = self._BeautifulSoup(text, "html.parser")

        if self._preserve_links:
            for a_tag in soup.find_all("a"):
                link_text = a_tag.get_text(strip=True)
                link_href = a_tag.get("href", "")
                markdown_link = f"[{link_text}]({link_href})"
                a_tag.replace_with(markdown_link)

        return self._process_html(soup)

    def _process_html(self, soup: Any) -> List[Document]:
        """
        Processes the HTML content using BeautifulSoup and splits it
        based on the headers.

        Args:
            soup (Any): Parsed HTML content using BeautifulSoup.

        Returns:
            List[Document]: A list of Document objects containing the split content.
        """
        documents: List[Document] = []
        current_headers: Dict[str, str] = {}
        current_content: List[str] = []
        preserved_elements: Dict[str, str] = {}
        placeholder_count: int = 0

        def _get_element_text(element: Any) -> str:
            if self._custom_handlers:
                handler = self._custom_handlers.get(element.name)
                if handler:
                    return handler(element)
            return element.get_text(separator=" ", strip=True) + " "

        elements = soup.find_all(recursive=False)

        def _process_element(
            element: List[Any],
            documents: List[Document],
            current_headers: Dict[str, str],
            current_content: List[str],
            preserved_elements: Dict[str, str],
            placeholder_count: int,
        ) -> Tuple[List[Document], Dict[str, str], List[str], Dict[str, str], int]:
            for elem in element:
                if elem.name.lower() in ["html", "body", "div"]:
                    children = elem.find_all(recursive=False)
                    (
                        documents,
                        current_headers,
                        current_content,
                        preserved_elements,
                        placeholder_count,
                    ) = _process_element(
                        children,
                        documents,
                        current_headers,
                        current_content,
                        preserved_elements,
                        placeholder_count,
                    )
                    continue

                if elem.name in [h[0] for h in self._headers_to_split_on]:
                    if current_content:
                        documents.extend(
                            self._create_documents(
                                current_headers,
                                " ".join(current_content),
                                preserved_elements,
                            )
                        )
                        current_content.clear()
                        preserved_elements.clear()
                    header_name = elem.get_text(strip=True)
                    current_headers = {
                        dict(self._headers_to_split_on)[elem.name]: header_name
                    }
                elif elem.name in self._elements_to_preserve:
                    placeholder = f"PRESERVED_{placeholder_count}"
                    preserved_elements[placeholder] = _get_element_text(elem)
                    current_content.append(placeholder)
                    placeholder_count += 1
                else:
                    current_content.append(_get_element_text(elem))

            return (
                documents,
                current_headers,
                current_content,
                preserved_elements,
                placeholder_count,
            )

        # Process the elements
        (
            documents,
            current_headers,
            current_content,
            preserved_elements,
            placeholder_count,
        ) = _process_element(
            elements,
            documents,
            current_headers,
            current_content,
            preserved_elements,
            placeholder_count,
        )

        # Handle any remaining content
        if current_content:
            documents.extend(
                self._create_documents(
                    current_headers, " ".join(current_content), preserved_elements
                )
            )

        return documents

    def _create_documents(
        self, headers: dict, content: str, preserved_elements: dict
    ) -> List[Document]:
        """
        Creates Document objects from the provided headers, content,
        and preserved elements.

        Args:
            headers (dict): The headers to attach as metadata to the Document.
            content (str): The content of the Document.
            preserved_elements (dict): Preserved elements to be reinserted
            into the content.

        Returns:
            List[Document]: A list of Document objects.
        """
        content = re.sub(r"\s+", " ", content).strip()

        if len(content) <= self._max_chunk_size:
            page_content = self._reinsert_preserved_elements(
                content, preserved_elements
            )
            return [Document(page_content=page_content, metadata=headers)]
        else:
            return self._further_split_chunk(content, headers, preserved_elements)

    def _further_split_chunk(
        self, content: str, metadata: dict, preserved_elements: dict
    ) -> List[Document]:
        """
        Further splits the content into smaller chunks
        if it exceeds the maximum chunk size.

        Args:
            content (str): The content to be split.
            metadata (dict): Metadata to attach to each chunk.
            preserved_elements (dict): Preserved elements
            to be reinserted into each chunk.

        Returns:
            List[Document]: A list of Document objects containing the split content.
        """
        splits = self._recursive_splitter.split_text(content)
        result = []

        for split in splits:
            split_with_preserved = self._reinsert_preserved_elements(
                split, preserved_elements
            )
            if split_with_preserved.strip():
                result.append(
                    Document(
                        page_content=split_with_preserved.strip(), metadata=metadata
                    )
                )

        return result

    def _reinsert_preserved_elements(
        self, content: str, preserved_elements: dict
    ) -> str:
        """
        Reinserts preserved elements into the content into their original positions.

        Args:
            content (str): The content where placeholders need to be replaced.
            preserved_elements (dict): Preserved elements to be reinserted.

        Returns:
            str: The content with placeholders replaced by preserved elements.
        """
        for placeholder, preserved_content in preserved_elements.items():
            content = content.replace(placeholder, preserved_content)
        return content
