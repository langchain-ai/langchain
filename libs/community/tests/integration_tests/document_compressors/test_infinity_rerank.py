from langchain_core.documents import Document

from langchain_community.document_compressors.infinity_rerank import (
    InfinityRerank,
)


def test_rerank() -> None:
    reranker = InfinityRerank()
    docs = [
        Document(
            page_content=(
                "This is a document not related to the python package infinity_emb, "
                "hence..."
            )
        ),
        Document(page_content="Paris is in France!"),
        Document(
            page_content=(
                "infinity_emb is a package for sentence embeddings and rerankings using"
                " transformer models in Python!"
            )
        ),
        Document(page_content="random text for nothing"),
    ]
    compressed = reranker.compress_documents(
        query="What is the python package infinity_emb?",
        documents=docs,
    )

    assert len(compressed) == 3, "default top_n is 3"
    assert compressed[0].page_content == docs[2].page_content, "rerank works"
