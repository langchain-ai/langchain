from typing import Any, Sequence

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)
from langchain_community.graph_vectorstores.links import copy_with_links


@beta()
class LinkExtractorTransformer(BaseDocumentTransformer):
    """DocumentTransformer for applying one or more LinkExtractors.

    Example:
        .. code-block:: python

            extract_links = LinkExtractorTransformer([
                HtmlLinkExtractor().as_document_extractor(),
            ])
            extract_links.transform_documents(docs)
    """

    def __init__(self, link_extractors: Sequence[LinkExtractor[Document]]):
        """Create a DocumentTransformer which adds extracted links to each document."""
        self.link_extractors = link_extractors

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Implement `transform_docments` directly, so that LinkExtractors which operate
        # better in batch (`extract_many`) get a chance to do so.

        # Run each extractor over all documents.
        links_per_extractor = [e.extract_many(documents) for e in self.link_extractors]

        # Transpose the list of lists to pair each document with the tuple of links.
        links_per_document = zip(*links_per_extractor)

        return [
            copy_with_links(document, *links)
            for document, links in zip(documents, links_per_document)
        ]
