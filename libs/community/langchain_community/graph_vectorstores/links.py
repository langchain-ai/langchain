from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Union

from langchain_core._api import beta
from langchain_core.documents import Document


@beta()
@dataclass(frozen=True)
class Link:
    """A link to/from a tag of a given kind.

    Documents in a :class:`graph vector store <langchain_community.graph_vectorstores.base.GraphVectorStore>`
    are connected via "links".
    Links form a bipartite graph between documents and tags: documents are connected
    to tags, and tags are connected to other documents.
    When documents are retrieved from a graph vector store, a pair of documents are
    connected with a depth of one if both documents are connected to the same tag.

    Links have a ``kind`` property, used to namespace different tag identifiers.
    For example a link to a keyword might use kind ``kw``, while a link to a URL might
    use kind ``url``.
    This allows the same tag value to be used in different contexts without causing
    name collisions.

    Links are directed. The directionality of links controls how the graph is
    traversed at retrieval time.
    For example, given documents ``A`` and ``B``, connected by links to tag ``T``:

    +----------+----------+---------------------------------+
    | A to T   | B to T   | Result                          |
    +==========+==========+=================================+
    | outgoing | incoming | Retrieval traverses from A to B |
    +----------+----------+---------------------------------+
    | incoming | incoming | No traversal from A to B        |
    +----------+----------+---------------------------------+
    | outgoing | incoming | No traversal from A to B        |
    +----------+----------+---------------------------------+
    | bidir    | incoming | Retrieval traverses from A to B |
    +----------+----------+---------------------------------+
    | bidir    | outgoing | No traversal from A to B        |
    +----------+----------+---------------------------------+
    | outgoing | bidir    | Retrieval traverses from A to B |
    +----------+----------+---------------------------------+
    | incoming | bidir    | No traversal from A to B        |
    +----------+----------+---------------------------------+

    Directed links make it possible to describe relationships such as term
    references / definitions: term definitions are generally relevant to any documents
    that use the term, but the full set of documents using a term generally aren't
    relevant to the term's definition.

    .. seealso::

        - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
        - :class:`How to link Documents on hyperlinks in HTML <langchain_community.graph_vectorstores.extractors.html_link_extractor.HtmlLinkExtractor>`
        - :class:`How to link Documents on common keywords (using KeyBERT) <langchain_community.graph_vectorstores.extractors.keybert_link_extractor.KeybertLinkExtractor>`
        - :class:`How to link Documents on common named entities (using GliNER) <langchain_community.graph_vectorstores.extractors.gliner_link_extractor.GLiNERLinkExtractor>`

    How to add links to a Document
    ==============================

    How to create links
    -------------------

    You can create links using the Link class's constructors :meth:`incoming`,
    :meth:`outgoing`, and :meth:`bidir`::

        from langchain_community.graph_vectorstores.links import Link

        print(Link.bidir(kind="location", tag="Paris"))

    .. code-block:: output

        Link(kind='location', direction='bidir', tag='Paris')

    Extending documents with links
    ------------------------------

    Now that we know how to create links, let's associate them with some documents.
    These edges will strengthen the connection between documents that share a keyword
    when using a graph vector store to retrieve documents.

    First, we'll load some text and chunk it into smaller pieces.
    Then we'll add a link to each document to link them all together::

        from langchain_community.document_loaders import TextLoader
        from langchain_community.graph_vectorstores.links import add_links
        from langchain_text_splitters import CharacterTextSplitter

        loader = TextLoader("state_of_the_union.txt")

        raw_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        for doc in documents:
            add_links(doc, Link.bidir(kind="genre", tag="oratory"))

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'state_of_the_union.txt', 'links': [Link(kind='genre', direction='bidir', tag='oratory')]}

    As we can see, each document's metadata now includes a bidirectional link to the
    genre ``oratory``.

    The documents can then be added to a graph vector store::

        from langchain_community.graph_vectorstores import CassandraGraphVectorStore

        graph_vectorstore = CassandraGraphVectorStore.from_documents(
            documents=documents, embeddings=...
        )

    """  # noqa: E501

    kind: str
    """The kind of link. Allows different extractors to use the same tag name without
    creating collisions between extractors. For example “keyword” vs “url”."""
    direction: Literal["in", "out", "bidir"]
    """The direction of the link."""
    tag: str
    """The tag of the link."""

    @staticmethod
    def incoming(kind: str, tag: str) -> "Link":
        """Create an incoming link.

        Args:
            kind: the link kind.
            tag: the link tag.
        """
        return Link(kind=kind, direction="in", tag=tag)

    @staticmethod
    def outgoing(kind: str, tag: str) -> "Link":
        """Create an outgoing link.

        Args:
            kind: the link kind.
            tag: the link tag.
        """
        return Link(kind=kind, direction="out", tag=tag)

    @staticmethod
    def bidir(kind: str, tag: str) -> "Link":
        """Create a bidirectional link.

        Args:
            kind: the link kind.
            tag: the link tag.
        """
        return Link(kind=kind, direction="bidir", tag=tag)


METADATA_LINKS_KEY = "links"


@beta()
def get_links(doc: Document) -> list[Link]:
    """Get the links from a document.

    Args:
        doc: The document to get the link tags from.
    Returns:
        The set of link tags from the document.
    """

    links = doc.metadata.setdefault(METADATA_LINKS_KEY, [])
    if not isinstance(links, list):
        # Convert to a list and remember that.
        links = list(links)
        doc.metadata[METADATA_LINKS_KEY] = links
    return links


@beta()
def add_links(doc: Document, *links: Union[Link, Iterable[Link]]) -> None:
    """Add links to the given metadata.

    Args:
        doc: The document to add the links to.
        *links: The links to add to the document.
    """
    links_in_metadata = get_links(doc)
    for link in links:
        if isinstance(link, Iterable):
            links_in_metadata.extend(link)
        else:
            links_in_metadata.append(link)


@beta()
def copy_with_links(doc: Document, *links: Union[Link, Iterable[Link]]) -> Document:
    """Return a document with the given links added.

    Args:
        doc: The document to add the links to.
        *links: The links to add to the document.

    Returns:
        A document with a shallow-copy of the metadata with the links added.
    """
    new_links = set(get_links(doc))
    for link in links:
        if isinstance(link, Iterable):
            new_links.update(link)
        else:
            new_links.add(link)

    return Document(
        page_content=doc.page_content,
        metadata={
            **doc.metadata,
            METADATA_LINKS_KEY: list(new_links),
        },
    )
