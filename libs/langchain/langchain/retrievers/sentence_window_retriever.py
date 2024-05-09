from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.retrievers import SentenceWindowRetriever

from langchain_community.retrievers.sentence_window_retriever import SentenceWindowRetriever

__all__ = ["SentenceWindowRetriever"]
