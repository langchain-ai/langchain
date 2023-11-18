"""Integration test for embedding-based redundant doc filtering."""
from langchain.document_transformers.embeddings_redundant_filter import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    _DocumentWithState,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


def test_embeddings_redundant_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "Where did all of my cookies go?",
        "I wish there were better Italian restaurants in my neighborhood.",
    ]
    docs = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    actual = redundant_filter.transform_documents(docs)
    assert len(actual) == 2
    assert set(texts[:2]).intersection([d.page_content for d in actual])


def test_embeddings_redundant_filter_with_state() -> None:
    texts = ["What happened to all of my cookies?", "foo bar baz"]
    state = {"embedded_doc": [0.5] * 10}
    docs = [_DocumentWithState(page_content=t, state=state) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    actual = redundant_filter.transform_documents(docs)
    assert len(actual) == 1


def test_embeddings_clustering_filter() -> None:
    texts = [
        "What happened to all of my cookies?",
        "A cookie is a small, baked sweet treat and you can find it in the cookie",
        "monsters' jar.",
        "Cookies are good.",
        "I have nightmares about the cookie monster.",
        "The most popular pizza styles are: Neapolitan, New York-style and",
        "Chicago-style. You can find them on iconic restaurants in major cities.",
        "Neapolitan pizza: This is the original pizza style,hailing from Naples,",
        "Italy.",
        "I wish there were better Italian Pizza restaurants in my neighborhood.",
        "New York-style pizza: This is characterized by its large, thin crust, and",
        "generous toppings.",
        "The first movie to feature a robot was 'A Trip to the Moon' (1902).",
        "The first movie to feature a robot that could pass for a human was",
        "'Blade Runner' (1982)",
        "The first movie to feature a robot that could fall in love with a human",
        "was 'Her' (2013)",
        "A robot is a machine capable of carrying out complex actions automatically.",
        "There are certainly hundreds, if not thousands movies about robots like:",
        "'Blade Runner', 'Her' and 'A Trip to the Moon'",
    ]

    docs = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    redundant_filter = EmbeddingsClusteringFilter(
        embeddings=embeddings,
        num_clusters=3,
        num_closest=1,
        sorted=True,
    )
    actual = redundant_filter.transform_documents(docs)
    assert len(actual) == 3
    assert texts[1] in [d.page_content for d in actual]
    assert texts[4] in [d.page_content for d in actual]
    assert texts[11] in [d.page_content for d in actual]
