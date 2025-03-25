from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Union
from urllib.parse import urldefrag, urljoin, urlparse

from langchain_core._api import beta
from langchain_core.documents import Document

from langchain_community.graph_vectorstores import Link
from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)
from langchain_community.graph_vectorstores.extractors.link_extractor_adapter import (
    LinkExtractorAdapter,
)

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import Tag


def _parse_url(link: Tag, page_url: str, drop_fragments: bool = True) -> Optional[str]:
    href = link.get("href")
    if href is None:
        return None
    url = urlparse(href)
    if url.scheme not in ["http", "https", ""]:
        return None

    # Join the HREF with the page_url to convert relative paths to absolute.
    url = str(urljoin(page_url, href))

    # Fragments would be useful if we chunked a page based on section.
    # Then, each chunk would have a different URL based on the fragment.
    # Since we aren't doing that yet, they just "break" links. So, drop
    # the fragment.
    if drop_fragments:
        return urldefrag(url).url
    return url


def _parse_hrefs(
    soup: BeautifulSoup, url: str, drop_fragments: bool = True
) -> Set[str]:
    soup_links: List[Tag] = soup.find_all("a")
    links: Set[str] = set()

    for link in soup_links:
        parse_url = _parse_url(link, page_url=url, drop_fragments=drop_fragments)
        # Remove self links and entries for any 'a' tag that failed to parse
        # (didn't have href, or invalid domain, etc.)
        if parse_url and parse_url != url:
            links.add(parse_url)

    return links


@dataclass
class HtmlInput:
    content: Union[str, BeautifulSoup]
    base_url: str


@beta()
class HtmlLinkExtractor(LinkExtractor[HtmlInput]):
    def __init__(self, *, kind: str = "hyperlink", drop_fragments: bool = True):
        """Extract hyperlinks from HTML content.

        Expects the input to be an HTML string or a `BeautifulSoup` object.

        Example::

            extractor = HtmlLinkExtractor()
            results = extractor.extract_one(HtmlInput(html, url))

        .. seealso::

            - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
            - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`

        How to link Documents on hyperlinks in HTML
        ===========================================

        Preliminaries
        -------------

        Install the ``beautifulsoup4`` package:

        .. code-block:: bash

            pip install -q langchain_community beautifulsoup4

        Usage
        -----

        For this example, we'll scrape 2 HTML pages that have an hyperlink from one
        page to the other using an ``AsyncHtmlLoader``.
        Then we use the ``HtmlLinkExtractor`` to create the links in the documents.

        Using extract_one()
        ^^^^^^^^^^^^^^^^^^^

        We can use :meth:`extract_one` on a document to get the links and add the links
        to the document metadata with
        :meth:`~langchain_community.graph_vectorstores.links.add_links`::

            from langchain_community.document_loaders import AsyncHtmlLoader
            from langchain_community.graph_vectorstores.extractors import (
                HtmlInput,
                HtmlLinkExtractor,
            )
            from langchain_community.graph_vectorstores.links import add_links
            from langchain_core.documents import Document

            loader = AsyncHtmlLoader(
                [
                    "https://python.langchain.com/docs/integrations/providers/astradb/",
                    "https://docs.datastax.com/en/astra/home/astra.html",
                ]
            )

            documents = loader.load()

            html_extractor = HtmlLinkExtractor()

            for doc in documents:
                links = html_extractor.extract_one(HtmlInput(doc.page_content, url))
                add_links(doc, links)

            documents[0].metadata["links"][:5]

        .. code-block:: output

            [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

        Using as_document_extractor()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        If you use a document loader that returns the raw HTML and that sets the source
        key in the document metadata such as ``AsyncHtmlLoader``,
        you can simplify by using :meth:`as_document_extractor` that takes directly a
        ``Document`` as input::

            from langchain_community.document_loaders import AsyncHtmlLoader
            from langchain_community.graph_vectorstores.extractors import HtmlLinkExtractor
            from langchain_community.graph_vectorstores.links import add_links

            loader = AsyncHtmlLoader(
                [
                    "https://python.langchain.com/docs/integrations/providers/astradb/",
                    "https://docs.datastax.com/en/astra/home/astra.html",
                ]
            )
            documents = loader.load()
            html_extractor = HtmlLinkExtractor().as_document_extractor()

            for document in documents:
                links = html_extractor.extract_one(document)
                add_links(document, links)

            documents[0].metadata["links"][:5]

        .. code-block:: output

            [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

        Using LinkExtractorTransformer
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Using the :class:`~langchain_community.graph_vectorstores.extractors.link_extractor_transformer.LinkExtractorTransformer`,
        we can simplify the link extraction::

            from langchain_community.document_loaders import AsyncHtmlLoader
            from langchain_community.graph_vectorstores.extractors import (
                HtmlLinkExtractor,
                LinkExtractorTransformer,
            )
            from langchain_community.graph_vectorstores.links import add_links

            loader = AsyncHtmlLoader(
                [
                    "https://python.langchain.com/docs/integrations/providers/astradb/",
                    "https://docs.datastax.com/en/astra/home/astra.html",
                ]
            )

            documents = loader.load()
            transformer = LinkExtractorTransformer([HtmlLinkExtractor().as_document_extractor()])
            documents = transformer.transform_documents(documents)

            documents[0].metadata["links"][:5]

        .. code-block:: output

            [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
             Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

        We can check that there is a link from the first document to the second::

            for doc_to in documents:
                for link_to in doc_to.metadata["links"]:
                    if link_to.direction == "in":
                        for doc_from in documents:
                            for link_from in doc_from.metadata["links"]:
                                if (
                                    link_to.direction == "in"
                                    and link_from.direction == "out"
                                    and link_to.tag == link_from.tag
                                ):
                                    print(
                                        f"Found link from {doc_from.metadata['source']} to {doc_to.metadata['source']}."
                                    )

        .. code-block:: output

            Found link from https://python.langchain.com/docs/integrations/providers/astradb/ to https://docs.datastax.com/en/astra/home/astra.html.

        The documents with URL links can then be added to a :class:`~langchain_community.graph_vectorstores.base.GraphVectorStore`::

            from langchain_community.graph_vectorstores import CassandraGraphVectorStore

            store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

        Args:
            kind: The kind of edge to extract. Defaults to ``hyperlink``.
            drop_fragments: Whether fragments in URLs and links should be
                dropped. Defaults to ``True``.
        """  # noqa: E501
        try:
            import bs4  # noqa:F401
        except ImportError as e:
            raise ImportError(
                "BeautifulSoup4 is required for HtmlLinkExtractor. "
                "Please install it with `pip install beautifulsoup4`."
            ) from e

        self._kind = kind
        self.drop_fragments = drop_fragments

    def as_document_extractor(
        self, url_metadata_key: str = "source"
    ) -> LinkExtractor[Document]:
        """Return a LinkExtractor that applies to documents.

        Note:
            Since the HtmlLinkExtractor parses HTML, if you use with other similar
            link extractors it may be more efficient to call the link extractors
            directly on the parsed BeautifulSoup object.

        Args:
            url_metadata_key: The name of the filed in document metadata with the URL of
                the document.
        """
        return LinkExtractorAdapter(
            underlying=self,
            transform=lambda doc: HtmlInput(
                doc.page_content, doc.metadata[url_metadata_key]
            ),
        )

    def extract_one(
        self,
        input: HtmlInput,  # noqa: A002
    ) -> Set[Link]:
        content = input.content
        if isinstance(content, str):
            from bs4 import BeautifulSoup

            content = BeautifulSoup(content, "html.parser")

        base_url = input.base_url
        if self.drop_fragments:
            base_url = urldefrag(base_url).url

        hrefs = _parse_hrefs(content, base_url, self.drop_fragments)

        links = {Link.outgoing(kind=self._kind, tag=url) for url in hrefs}
        links.add(Link.incoming(kind=self._kind, tag=base_url))
        return links
