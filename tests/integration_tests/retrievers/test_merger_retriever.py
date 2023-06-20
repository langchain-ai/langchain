from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import Chroma


def test_merger_retriever_get_relevant_docs() -> None:
    """Test get_relevant_docs."""
    texts_group_a = [
        "This is a document about the Boston Celtics",
        "Fly me to the moon is one of my favourite songs."
        "I simply love going to the movies",
    ]
    texts_group_b = [
        "This is a document about the Poenix Suns",
        "The Boston Celtics won the game by 20 points",
        "Real stupidity beats artificial intelligence every time. TP",
    ]
    embeddings = OpenAIEmbeddings()
    retriever_a = Chroma.from_texts(texts_group_a, embedding=embeddings).as_retriever(
        search_kwargs={"k": 1}
    )
    retriever_b = Chroma.from_texts(texts_group_b, embedding=embeddings).as_retriever(
        search_kwargs={"k": 1}
    )

    # The Lord of the Retrievers.
    lotr = MergerRetriever([retriever_a, retriever_b])

    actual = lotr.get_relevant_documents("Tell me about the Celtics")
    assert len(actual) == 2
    assert texts_group_a[0] in [d.page_content for d in actual]
    assert texts_group_b[1] in [d.page_content for d in actual]
