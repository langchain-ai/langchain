"""Question-answering with sources over an index."""

from ..qa_with_references.retrieval import RetrievalQAWithReferencesChain
from .base import (
    BaseQAWithReferencesAndVerbatimsChain,
)


class RetrievalQAWithReferencesAndVerbatimsChain(
    RetrievalQAWithReferencesChain, BaseQAWithReferencesAndVerbatimsChain
):
    """Question-answering with referenced documents and verbatim.

    This implementation allows you to retrieve the list of documents, enriched with
    the verbatim metadata.
    The implementation uses fewer tokens and correctly handles recursive map_reduces.
    """
