from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, cast

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

class ElementType(TypedDict):
    """Element type as typed dict."""

    url: str
    xpath: str
    content: str
    metadata: Dict[str, str]



@dataclass
class Node:
    """
    Represents a node in a hierarchical structure.

    Attributes:
        name: The name of the node.
        tag_type: The type of the node.
        content: The content of the node.
        level: The level of the node in the hierarchy.
        dom_depth: The depth of the node in the DOM structure.
        parent: The parent node. Defaults to None.
    """
    name: str
    tag_type: str
    content: str
    level: int
    dom_depth: int
    parent: Optional[Node] = field(default=None)

class HTMLHeaderTextSplitter:
    """
    Splits HTML content into structured `Document` objects based on specified header
    tags.

    This splitter processes HTML by identifying header elements (e.g., `<h1>`,
    `<h2>`) and segments the content accordingly. Each header and the text that
    follows, up to the next header of the same or higher level, are grouped into a
    `Document`. The metadata of each `Document` reflects the hierarchy of headers,
    providing an organized content structure.

    If the content does not contain any of the specified headers, the splitter
    returns a single `Document` with the aggregated content and no additional
    metadata.
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_element: bool = False
    ) -> None:
        """
        Initialize with headers to split on.

        Args:
            headers_to_split_on: A list of tuples where
                each tuple contains a header tag and its corresponding value.
            return_each_element: Whether to return each HTML
                element as a separate Document. Defaults to False.
        """
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda x: int(x[0][1])
        )
        self.header_mapping = dict(self.headers_to_split_on)
        self.header_tags = [tag for tag, _ in self.headers_to_split_on]
        self.elements_tree: Dict[int, Tuple[str, str, int, int]] = {}
        self.return_each_element = return_each_element

    def _header_level(self, element) -> int:
        """
        Determine the heading level of an element.

        Args:
            element: A BeautifulSoup element.

        Returns:
            The heading level (1-6) if a heading, else a large number.
        """
        tag_name = element.name.lower() if hasattr(element, 'name') else ''
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return int(tag_name[1])
        return 9999

    def _dom_depth(self, element) -> int:
        """
        Compute the DOM depth of an element.

        Args:
            element: A BeautifulSoup element.

        Returns:
            The depth of the element in the DOM tree.
        """
        depth = 0
        for _ in element.parents:
            depth += 1
        return depth

    def _build_tree(self, elements) -> None:
        """
        Build a tree structure from a list of HTML elements.

        Args:
            elements: A list of BeautifulSoup elements.
        """
        for idx, element in enumerate(elements):
            text = ' '.join(
                t for t in element.find_all(string=True, recursive=False)
                if isinstance(t, str)
            ).strip()

            if not text:
                continue

            level = self._header_level(element)
            dom_depth = self._dom_depth(element)

            self.elements_tree[idx] = (
                element.name,
                text,
                level,
                dom_depth
            )

    def split_text(self, text: str) -> List[Document]:
        """
        Split the given text into a list of Document objects.

        Args:
            text: The HTML text to split.

        Returns:
            A list of split Document objects.
        """
        return self.split_text_from_file(StringIO(text))

    def split_text_from_url(
        self,
        url: str,
        timeout: int = 10,
        **kwargs: Any
    ) -> List[Document]:
        """
        Fetch text content from a URL and split it into documents.

        Args:
            url: The URL to fetch content from.
            timeout: Timeout for the request. Defaults to 10.
            **kwargs: Additional keyword arguments for the request.

        Returns:
            A list of split Document objects.

        Raises:
            requests.RequestException: If the HTTP request fails.
        """
        try:
            kwargs.setdefault('timeout', timeout)
            response = requests.get(url, **kwargs)  # noqa: E501
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            raise e
        return self.split_text_from_file(BytesIO(response.content))

    def _finalize_chunk(
        self,
        current_chunk: List[str],
        active_headers: Dict[str, Tuple[str, int, int]],
        documents: List[Document],
        chunk_dom_depth: int) -> None:

        if current_chunk:
            final_meta: Dict[str, str] = {
                key: content for key, (content, level, dom_depth) in active_headers.items()
                if chunk_dom_depth >= dom_depth
            }
            combined_text = "  \n".join(
                line for line in current_chunk if line.strip()
            )
            documents.append(
                Document(page_content=combined_text, metadata=final_meta)
            )
            current_chunk.clear()
            chunk_dom_depth = 0


    def _generate_documents(self, nodes: Dict[int, Node]) -> List[Document]:
        """
        Generate a list of Document objects from a node structure.

        Args:
            A dictionary of nodes indexed by their position.

        Returns:
            A list of generated Document objects.
        """
        documents: List[Document] = []
        active_headers: Dict[str, Tuple[str, int, int]] = {}
        current_chunk: List[str] = []
        chunk_dom_depth = 0



        def process_node(node: Node) -> None:
            """
            Processes a given node and updates the current chunk, active headers, and
            documents based on the node's type and content.
            Args:
                node: The node to be processed. It should have attributes
                    'tag_type', 'content', 'level', and 'dom_depth'.
            """

            nonlocal chunk_dom_depth
            node_type = node.tag_type  # type: ignore[attr-defined]
            node_content = node.content  # type: ignore[attr-defined]
            node_level = node.level  # type: ignore[attr-defined]
            node_dom_depth = node.dom_depth  # type: ignore[attr-defined]

            if node_type in self.header_tags:
                self._finalize_chunk(current_chunk, active_headers, documents, chunk_dom_depth)
                headers_to_remove = [
                    key for key, (_, lvl, _) in active_headers.items()
                    if lvl >= node_level
                ]
                for key in headers_to_remove:
                    del active_headers[key]
                header_key = self.header_mapping[node_type]  # type: ignore[attr-defined]
                active_headers[header_key] = (
                    node_content,
                    node_level,
                    node_dom_depth
                )
                header_meta: Dict[str, str] = {
                    key: content for key, (content, lvl, dd) in active_headers.items()
                    if node_dom_depth >= dd
                }
                documents.append(
                    Document(
                        page_content=node_content,
                        metadata=header_meta
                    )
                )
            else:
                headers_to_remove = [
                    key for key, (_, _, dd) in active_headers.items()
                    if node_dom_depth < dd
                ]
                for key in headers_to_remove:
                    del active_headers[key]
                if node_content.strip():
                    current_chunk.append(node_content)
                    chunk_dom_depth = max(chunk_dom_depth, node_dom_depth)

        sorted_nodes = sorted(nodes.items())
        for _, node in sorted_nodes:
            process_node(node)

        self._finalize_chunk(current_chunk, active_headers, documents, chunk_dom_depth)
        return documents

    def split_text_from_file(self, file: Any) -> List[Document]:
        """
        Split HTML content from a file into a list of Document objects.

        Args:
            file: A file path or a file-like object containing HTML content.

        Returns:
            A list of split Document objects.
        """
        if isinstance(file, str):
            with open(file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        else:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        body = soup.body if soup.body else soup

        elements = body.find_all()
        self._build_tree(elements)

        if not self.elements_tree:
            return []

        min_level = min(
            level for (_, _, level, _) in self.elements_tree.values()
        )
        root = Node(
            "root",
            tag_type="root",
            content="",
            level=min_level - 1,
            dom_depth=0
        )

        nodes = {
            idx: Node(
                f"{tag}_{idx}",
                tag_type=tag,
                content=text,
                level=level,
                dom_depth=dom_depth
            )
            for idx, (tag, text, level, dom_depth) in self.elements_tree.items()
        }

        stack = []
        for idx in sorted(nodes):
            node = nodes[idx]
            while stack and (
                stack[-1].level >= node.level
                or stack[-1].dom_depth >= node.dom_depth
                ):
                stack.pop()
            if stack:
                node.parent = stack[-1]
            else:
                node.parent = root
            stack.append(node)

        if not self.return_each_element:
            return self._aggregate_documents(nodes)

        return self._generate_individual_documents(nodes)

    def _aggregate_documents(self, nodes: Dict[int, Node]) -> List[Document]:
        """
        Aggregate documents based on headers.

        Args:
            nodes: A dictionary of nodes indexed by their position.

        Returns:
            A list of aggregated Document objects.
        """
        # Reuse the existing _generate_documents method for aggregation
        return self._generate_documents(nodes)

    def _generate_individual_documents(self, nodes: Dict[int, Node]) -> List[Document]:
        """
        Generate individual Document objects for each element.

        Args:
            nodes: A dictionary of nodes indexed by their position.

        Returns:
            A list of individual Document objects.
        """
        documents: List[Document] = []
        active_headers: Dict[str, Tuple[str, int, int]] = {}

        sorted_nodes = sorted(nodes.items())

        def process_node(node: Node) -> None:
            """
            Process a single node to create Document objects based on header tags.

            Args:
                node: The node to process.
            """
            node_type = node.type  # type: ignore[attr-defined]
            node_content = node.content  # type: ignore[attr-defined]
            node_level = node.level  # type: ignore[attr-defined]
            node_dom_depth = node.dom_depth  # type: ignore[attr-defined]

            if node_type in self.header_tags:
                # Remove headers of the same or lower level
                headers_to_remove = [
                    key for key, (_, lvl, _) in active_headers.items()
                    if lvl >= node_level
                ]
                for key in headers_to_remove:
                    del active_headers[key]

                # Update active headers with the current header
                header_key = self.header_mapping[node_type]  # type: ignore[attr-defined]
                active_headers[header_key] = (
                    node_content,
                    node_level,
                    node_dom_depth
                )

                # Create metadata based on active headers
                header_meta: Dict[str, str] = {
                    key: content for key, (content, lvl, dd) in active_headers.items()
                    if node_dom_depth >= dd
                }

                # Create a Document for the header element
                documents.append(
                    Document(
                        page_content=node_content,
                        metadata=header_meta
                    )
                )
            else:
                # For non-header elements, associate with current headers
                if node_content.strip():
                    header_meta: Dict[str, str] = {
                        key: content for key, (content, lvl, dd) in active_headers.items()
                        if node_dom_depth >= dd
                    }
                    documents.append(
                        Document(
                            page_content=node_content,
                            metadata=header_meta
                        )
                    )

        # Process each node using the inner process_node function
        for _, node in sorted_nodes:
            process_node(node)

        return documents


class HTMLSectionSplitter:
    """Splitting HTML files based on specified tag and font sizes.

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
            **kwargs (Any): Additional optional arguments for customizations.

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
        """Split HTML text string.

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
        """Split an HTML document into sections based on specified header tags.

        This method uses BeautifulSoup to parse the HTML content and divides it into
        sections based on headers defined in `headers_to_split_on`. Each section
        contains the header text, content under the header, and the tag name.

        Args:
            html_doc (str): The HTML document to be split into sections.

        Returns:
            List[Dict[str, Optional[str]]]: A list of dictionaries representing
            sections.
                Each dictionary contains:
                - 'header': The header text or a default title for the first section.
                - 'content': The content under the header.
                - 'tag_name': The name of the header tag (e.g., "h1", "h2").
        """
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
                current_header_tag = header_element.name  # type: ignore[attr-defined]
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
        """Convert specific HTML tags to headers using an XSLT transformation.

        This method uses an XSLT file to transform the HTML content, converting
        certain tags into headers for easier parsing. If no XSLT path is provided,
        the HTML content is returned unchanged.

        Args:
            html_content (str): The HTML content to be transformed.

        Returns:
            str: The transformed HTML content as a string.
        """
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
        """Split HTML file.

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
