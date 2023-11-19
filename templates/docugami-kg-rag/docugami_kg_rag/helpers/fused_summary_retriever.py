from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.vectorstore import VectorStore


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


@dataclass
class FusedDocumentElements:
    rank: int
    summary: str
    fragments: List[Document]
    source: str


DOCUMENT_SUMMARY_TEMPLATE: str = """
--------------------------------
**** DOCUMENT NAME: {doc_name}
**** DOCUMENT SUMMARY:
{summary}
**** RELEVANT FRAGMENTS:
{fragments}
--------------------------------
"""


class FusedSummaryRetriever(BaseRetriever):
    """
    Retrieve a fused document that consist of pre-calculated summaries.
    """

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors."""

    summarystore: BaseStore[str, str]
    """The storage layer for the parent document summaries."""

    parent_id_key: str = "doc_id"
    """Metadata key for parent doc ID."""

    source_key: str = "source"
    """Metadata key for source."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        fused_doc_elements: Dict[str, FusedDocumentElements] = {}
        for i in range(len(sub_docs)):
            sub_doc = sub_docs[i]
            parent_id = sub_doc.metadata.get(self.parent_id_key)
            if parent_id:
                summaries = self.summarystore.mget([parent_id])
                summary: str = ""
                if summaries:
                    summary: str = summaries[0]  # type: ignore
                if not summary:
                    raise Exception(
                        f"No summary found for {parent_id} in summary store, please pre-load summaries for all parents."
                    )
                source = sub_doc.metadata.get(self.source_key)
                if not source:
                    raise Exception(f"No name found in metadata for: {sub_doc}.")

                if parent_id not in fused_doc_elements:
                    # Init fused parent with information from most relevant sub-doc
                    fused_doc_elements[parent_id] = FusedDocumentElements(
                        rank=i,
                        summary=summary,
                        fragments=[sub_doc],
                        source=source,
                    )
                else:
                    fused_doc_elements[parent_id].fragments.append(sub_doc)

        fused_docs: List[Document] = []
        for element in sorted(fused_doc_elements.values(), key=lambda x: x.rank):
            fragments_str = "\n\n".join([d.page_content.strip() for d in element.fragments])
            fused_docs.append(
                Document(
                    page_content=DOCUMENT_SUMMARY_TEMPLATE.format(
                        doc_name=element.source,
                        summary=element.summary,
                        fragments=fragments_str,
                    )
                )
            )

        return fused_docs
