from typing import Callable, List, Set

from langchain_core._api import beta
from langchain_core.documents import Document

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)
from langchain_community.graph_vectorstores.extractors.link_extractor_adapter import (
    LinkExtractorAdapter,
)
from langchain_community.graph_vectorstores.links import Link

# TypeAlias is not available in Python 3.9, we can't use that or the newer `type`.
HierarchyInput = List[str]

_PARENT: str = "p:"
_CHILD: str = "c:"
_SIBLING: str = "s:"


@beta()
class HierarchyLinkExtractor(LinkExtractor[HierarchyInput]):
    def __init__(
        self,
        *,
        kind: str = "hierarchy",
        parent_links: bool = True,
        child_links: bool = False,
        sibling_links: bool = False,
    ):
        """Extract links from a document hierarchy.

        Example:

            .. code-block:: python

                # Given three paths (in this case, within the "Root" document):
                h1 = ["Root", "H1"]
                h1a = ["Root", "H1", "a"]
                h1b = ["Root", "H1", "b"]

                # Parent links `h1a` and `h1b` to `h1`.
                # Child links `h1` to `h1a` and `h1b`.
                # Sibling links `h1a` and `h1b` together (both directions).

        Example use with documents:
            .. code_block: python
                transformer = LinkExtractorTransformer([
                    HierarchyLinkExtractor().as_document_extractor(
                        # Assumes the "path" to each document is in the metadata.
                        # Could split strings, etc.
                        lambda doc: doc.metadata.get("path", [])
                    )
                ])
                linked = transformer.transform_documents(docs)

        Args:
            kind: Kind of links to produce with this extractor.
            parent_links: Link from a section to its parent.
            child_links: Link from a section to its children.
            sibling_links: Link from a section to other sections with the same parent.
        """
        self._kind = kind
        self._parent_links = parent_links
        self._child_links = child_links
        self._sibling_links = sibling_links

    def as_document_extractor(
        self, hierarchy: Callable[[Document], HierarchyInput]
    ) -> LinkExtractor[Document]:
        """Create a LinkExtractor from `Document`.

        Args:
            hierarchy: Function that returns the path for the given document.

        Returns:
            A `LinkExtractor[Document]` suitable for application to `Documents` directly
            or with `LinkExtractorTransformer`.
        """
        return LinkExtractorAdapter(underlying=self, transform=hierarchy)

    def extract_one(
        self,
        input: HierarchyInput,
    ) -> Set[Link]:
        this_path = "/".join(input)
        parent_path = None

        links = set()
        if self._parent_links:
            # This is linked from everything with this parent path.
            links.add(Link.incoming(kind=self._kind, tag=_PARENT + this_path))
        if self._child_links:
            # This is linked to every child with this as it's "parent" path.
            links.add(Link.outgoing(kind=self._kind, tag=_CHILD + this_path))

        if len(input) >= 1:
            parent_path = "/".join(input[0:-1])
            if self._parent_links and len(input) > 1:
                # This is linked to the nodes with the given parent path.
                links.add(Link.outgoing(kind=self._kind, tag=_PARENT + parent_path))
            if self._child_links and len(input) > 1:
                # This is linked from every node with the given parent path.
                links.add(Link.incoming(kind=self._kind, tag=_CHILD + parent_path))
            if self._sibling_links:
                # This is a sibling of everything with the same parent.
                links.add(Link.bidir(kind=self._kind, tag=_SIBLING + parent_path))

        return links
