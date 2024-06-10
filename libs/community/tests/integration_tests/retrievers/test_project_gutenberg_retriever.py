"""Test Project Gutenberg retriever."""
from langchain_core.documents import Document

from langchain_community.retrievers import ProjectGutenbergRetriever


def test_project_gutenberg_retriever() -> None:

    docai_wh_retriever = ProjectGutenbergRetriever()
    documents = docai_wh_retriever.invoke(
        "Crime and Punishment"
    )
    assert len(documents) == 1
    for doc in documents:
        assert isinstance(doc, Document)
