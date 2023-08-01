"""Integration test for doc reordering."""
from langchain.document_transformers.long_context_reorder import LongContextReorder
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def test_long_context_reorder() -> None:
    """Test Lost in the middle reordering get_relevant_docs."""
    texts = [
        "Basquetball is a great sport.",
        "Fly me to the moon is one of my favourite songs.",
        "The Celtics are my favourite team.",
        "This is a document about the Boston Celtics",
        "I simply love going to the movies",
        "The Boston Celtics won the game by 20 points",
        "This is just a random text.",
        "Elden Ring is one of the best games in the last 15 years.",
        "L. Kornet is one of the best Celtics players.",
        "Larry Bird was an iconic NBA player.",
    ]
    embeddings = OpenAIEmbeddings()
    retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": 10}
    )
    reordering = LongContextReorder()
    docs = retriever.get_relevant_documents("Tell me about the Celtics")
    actual = reordering.transform_documents(docs)

    # First 2 and Last 2 elements must contain the most relevant
    first_and_last = list(actual[:2]) + list(actual[-2:])
    assert len(actual) == 10
    assert texts[2] in [d.page_content for d in first_and_last]
    assert texts[3] in [d.page_content for d in first_and_last]
    assert texts[5] in [d.page_content for d in first_and_last]
    assert texts[8] in [d.page_content for d in first_and_last]
