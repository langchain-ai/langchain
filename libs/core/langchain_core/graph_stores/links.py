from dataclasses import dataclass
from typing import Iterable, Literal, Set, Union

from langchain_core.documents import Document


@dataclass(frozen=True)
class Link:
    kind: str
    direction: Literal["in", "out", "bidir"]
    tag: str

    @staticmethod
    def incoming(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="in", tag=tag)

    @staticmethod
    def outgoing(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="out", tag=tag)

    @staticmethod
    def bidir(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="bidir", tag=tag)


METADATA_LINKS_KEY = "links"


def get_links(doc: Document) -> Set[Link]:
    """Get the links from a document.
    Args:
        doc: The document to get the link tags from.
    Returns:
        The set of link tags from the document.
    """

    links = doc.metadata.setdefault(METADATA_LINKS_KEY, set())
    if not isinstance(links, Set):
        # Convert to a set and remember that.
        links = set(links)
        doc.metadata[METADATA_LINKS_KEY] = links
    return links


def add_links(doc: Document, *links: Union[Link, Iterable[Link]]) -> None:
    """Add links to the given metadata.
    Args:
        doc: The document to add the links to.
        *links: The links to add to the document.
    """
    doc_links = get_links(doc)
    for link in links:
        if isinstance(link, Iterable):
            doc_links.update(link)
        else:
            doc_links.add(link)
