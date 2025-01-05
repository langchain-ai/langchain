from __future__ import annotations

import copy
import pathlib
import re
from io import BytesIO, StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import requests
from langchain_core._api import beta
from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class ElementType(TypedDict):
    """Element type as typed dict."""

    url: str
    xpath: str
    content: str
    metadata: Dict[str, str]


class HTMLHeaderTextSplitter:
    """Splitting HTML files based on specified headers.

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
        """Combine elements with common metadata into chunks.

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
        """Split HTML from web URL.

        Args:
            url: web URL
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the fetch url content request.
        """
        r = requests.get(url, **kwargs)
        return self.split_text_from_file(BytesIO(r.content))

    def split_text(self, text: str) -> List[Document]:
        """Split HTML text string.

        Args:
            text: HTML text
        """
        return self.split_text_from_file(StringIO(text))

    def split_text_from_file(self, file: Any) -> List[Document]:
        """Split HTML file.

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


@beta()
class HTMLSemanticPreservingSplitter(BaseDocumentTransformer):
    """Split HTML content preserving semantic structure.

    Splits HTML content by headers into generalized chunks, preserving semantic
    structure. If chunks exceed the maximum chunk size, it uses
    RecursiveCharacterTextSplitter for further splitting.

    The splitter preserves full HTML elements (e.g., <table>, <ul>) and converts
    links to Markdown-like links. It can also preserve images, videos, and audio
    elements by converting them into Markdown format. Note that some chunks may
    exceed the maximum size to maintain semantic integrity.

    .. versionadded: 0.3.5

    Args:
        headers_to_split_on (List[Tuple[str, str]]): HTML headers (e.g., "h1", "h2")
            that define content sections.
        max_chunk_size (int): Maximum size for each chunk, with allowance for
            exceeding this limit to preserve semantics.
        chunk_overlap (int): Number of characters to overlap between chunks to ensure
            contextual continuity.
        separators (List[str]): Delimiters used by RecursiveCharacterTextSplitter for
            further splitting.
        elements_to_preserve (List[str]): HTML tags (e.g., <table>, <ul>) to remain
            intact during splitting.
        preserve_links (bool): Converts <a> tags to Markdown links ([text](url)).
        preserve_images (bool): Converts <img> tags to Markdown images (![alt](src)).
        preserve_videos (bool): Converts <video> tags to Markdown
        video links (![video](src)).
        preserve_audio (bool): Converts <audio> tags to Markdown
        audio links (![audio](src)).
        custom_handlers (Dict[str, Callable[[Any], str]]): Optional custom handlers for
            specific HTML tags, allowing tailored extraction or processing.
        stopword_removal (bool): Optionally remove stopwords from the text.
        stopword_lang (str): The language of stopwords to remove.
        normalize_text (bool): Optionally normalize text
            (e.g., lowercasing, removing punctuation).
        external_metadata (Optional[Dict[str, str]]): Additional metadata to attach to
            the Document objects.
        allowlist_tags (Optional[List[str]]): Only these tags will be retained in
            the HTML.
        denylist_tags (Optional[List[str]]): These tags will be removed from the HTML.
        preserve_parent_metadata (bool): Whether to pass through parent document
            metadata to split documents when calling
            ``transform_documents/atransform_documents()``.

    Example:
        .. code-block:: python

            from langchain_text_splitters.html import HTMLSemanticPreservingSplitter

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
                preserve_images=True,
                custom_handlers={"iframe": custom_iframe_extractor}
            )
    """  # noqa: E501, D214

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        *,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 0,
        separators: Optional[List[str]] = None,
        elements_to_preserve: Optional[List[str]] = None,
        preserve_links: bool = False,
        preserve_images: bool = False,
        preserve_videos: bool = False,
        preserve_audio: bool = False,
        custom_handlers: Optional[Dict[str, Callable[[Any], str]]] = None,
        stopword_removal: bool = False,
        stopword_lang: str = "english",
        normalize_text: bool = False,
        external_metadata: Optional[Dict[str, str]] = None,
        allowlist_tags: Optional[List[str]] = None,
        denylist_tags: Optional[List[str]] = None,
        preserve_parent_metadata: bool = False,
    ):
        """Initialize splitter."""
        try:
            from bs4 import BeautifulSoup, Tag

            self._BeautifulSoup = BeautifulSoup
            self._Tag = Tag
        except ImportError:
            raise ImportError(
                "Could not import BeautifulSoup. "
                "Please install it with 'pip install bs4'."
            )

        self._headers_to_split_on = sorted(headers_to_split_on)
        self._max_chunk_size = max_chunk_size
        self._elements_to_preserve = elements_to_preserve or []
        self._preserve_links = preserve_links
        self._preserve_images = preserve_images
        self._preserve_videos = preserve_videos
        self._preserve_audio = preserve_audio
        self._custom_handlers = custom_handlers or {}
        self._stopword_removal = stopword_removal
        self._stopword_lang = stopword_lang
        self._normalize_text = normalize_text
        self._external_metadata = external_metadata or {}
        self._allowlist_tags = allowlist_tags
        self._preserve_parent_metadata = preserve_parent_metadata
        if allowlist_tags:
            self._allowlist_tags = list(
                set(allowlist_tags + [header[0] for header in headers_to_split_on])
            )
        self._denylist_tags = denylist_tags
        if denylist_tags:
            self._denylist_tags = [
                tag
                for tag in denylist_tags
                if tag not in [header[0] for header in headers_to_split_on]
            ]
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

        if self._stopword_removal:
            try:
                import nltk  # type: ignore
                from nltk.corpus import stopwords  # type: ignore

                nltk.download("stopwords")
                self._stopwords = set(stopwords.words(self._stopword_lang))
            except ImportError:
                raise ImportError(
                    "Could not import nltk. Please install it with 'pip install nltk'."
                )

    def split_text(self, text: str) -> List[Document]:
        """Splits the provided HTML text into smaller chunks based on the configuration.

        Args:
            text (str): The HTML content to be split.

        Returns:
            List[Document]: A list of Document objects containing the split content.
        """
        soup = self._BeautifulSoup(text, "html.parser")

        self._process_media(soup)

        if self._preserve_links:
            self._process_links(soup)

        if self._allowlist_tags or self._denylist_tags:
            self._filter_tags(soup)

        return self._process_html(soup)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> List[Document]:
        """Transform sequence of documents by splitting them."""
        transformed = []
        for doc in documents:
            splits = self.split_text(doc.page_content)
            if self._preserve_parent_metadata:
                splits = [
                    Document(
                        page_content=split_doc.page_content,
                        metadata={**doc.metadata, **split_doc.metadata},
                    )
                    for split_doc in splits
                ]
            transformed.extend(splits)
        return transformed

    def _process_media(self, soup: Any) -> None:
        """Processes the media elements.

        Process elements in the HTML content by wrapping them in a <media-wrapper> tag
        and converting them to Markdown format.

        Args:
            soup (Any): Parsed HTML content using BeautifulSoup.
        """
        if self._preserve_images:
            for img_tag in soup.find_all("img"):
                img_src = img_tag.get("src", "")
                markdown_img = f"![image:{img_src}]({img_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_img
                img_tag.replace_with(wrapper)

        if self._preserve_videos:
            for video_tag in soup.find_all("video"):
                video_src = video_tag.get("src", "")
                markdown_video = f"![video:{video_src}]({video_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_video
                video_tag.replace_with(wrapper)

        if self._preserve_audio:
            for audio_tag in soup.find_all("audio"):
                audio_src = audio_tag.get("src", "")
                markdown_audio = f"![audio:{audio_src}]({audio_src})"
                wrapper = soup.new_tag("media-wrapper")
                wrapper.string = markdown_audio
                audio_tag.replace_with(wrapper)

    def _process_links(self, soup: Any) -> None:
        """Processes the links in the HTML content.

        Args:
            soup (Any): Parsed HTML content using BeautifulSoup.
        """
        for a_tag in soup.find_all("a"):
            a_href = a_tag.get("href", "")
            a_text = a_tag.get_text(strip=True)
            markdown_link = f"[{a_text}]({a_href})"
            wrapper = soup.new_tag("link-wrapper")
            wrapper.string = markdown_link
            a_tag.replace_with(markdown_link)

    def _filter_tags(self, soup: Any) -> None:
        """Filters the HTML content based on the allowlist and denylist tags.

        Args:
            soup (Any): Parsed HTML content using BeautifulSoup.
        """
        if self._allowlist_tags:
            for tag in soup.find_all(True):
                if tag.name not in self._allowlist_tags:
                    tag.decompose()

        if self._denylist_tags:
            for tag in soup.find_all(self._denylist_tags):
                tag.decompose()

    def _normalize_and_clean_text(self, text: str) -> str:
        """Normalizes the text by removing extra spaces and newlines.

        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text.
        """
        if self._normalize_text:
            text = text.lower()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        if self._stopword_removal:
            text = " ".join(
                [word for word in text.split() if word not in self._stopwords]
            )

        return text

    def _process_html(self, soup: Any) -> List[Document]:
        """Processes the HTML content using BeautifulSoup and splits it using headers.

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
            """Recursively extracts and processes the text of an element.

            Applies custom handlers where applicable, and ensures correct spacing.

            Args:
                element (Any): The HTML element to process.

            Returns:
                str: The processed text of the element.
            """
            if element.name in self._custom_handlers:
                return self._custom_handlers[element.name](element)

            text = ""

            if element.name is not None:
                for child in element.children:
                    child_text = _get_element_text(child).strip()
                    if text and child_text:
                        text += " "
                    text += child_text
            elif element.string:
                text += element.string

            return self._normalize_and_clean_text(text)

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
                    content = _get_element_text(elem)
                    if content:
                        current_content.append(content)

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
        """Creates Document objects from the provided headers, content, and elements.

        Args:
            headers (dict): The headers to attach as metadata to the Document.
            content (str): The content of the Document.
            preserved_elements (dict): Preserved elements to be reinserted
            into the content.

        Returns:
            List[Document]: A list of Document objects.
        """
        content = re.sub(r"\s+", " ", content).strip()

        metadata = {**headers, **self._external_metadata}

        if len(content) <= self._max_chunk_size:
            page_content = self._reinsert_preserved_elements(
                content, preserved_elements
            )
            return [Document(page_content=page_content, metadata=metadata)]
        else:
            return self._further_split_chunk(content, metadata, preserved_elements)

    def _further_split_chunk(
        self, content: str, metadata: dict, preserved_elements: dict
    ) -> List[Document]:
        """Further splits the content into smaller chunks.

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
        """Reinserts preserved elements into the content into their original positions.

        Args:
            content (str): The content where placeholders need to be replaced.
            preserved_elements (dict): Preserved elements to be reinserted.

        Returns:
            str: The content with placeholders replaced by preserved elements.
        """
        for placeholder, preserved_content in preserved_elements.items():
            content = content.replace(placeholder, preserved_content.strip())
        return content
