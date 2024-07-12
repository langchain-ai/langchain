from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer
from langchain_core.graph_vectorstores.links import add_links

from langchain_community.graph_vectorstores.extractors.link_extractor import (
    LinkExtractor,
)


class LinkExtractorTransformer(BaseDocumentTransformer):
    """DocumentTransformer for applying one or more LinkExtractors.

    Example:
        .. code-block:: python

            extract_links = LinkExtractorTransformer([
                HtmlLinkExtractor().as_document_extractor(),
            ])
            extract_links.transform_documents(docs)
    """

    def __init__(self, link_extractors: Iterable[LinkExtractor[Document]]):
        """Create a DocumentTransformer which adds extracted links to each document."""
        self.link_extractors = link_extractors

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Implement `transform_docments` directly, so that LinkExtractors which operate
        # better in batch (`extract_many`) get a chance to do so.
        document_links = zip(
            documents,
            zip(
                *[
                    extractor.extract_many(documents)
                    for extractor in self.link_extractors
                ]
            ),
        )
        for document, links in document_links:
            add_links(document, *links)
        return documents
