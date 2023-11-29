"""Test Vertex AI API wrapper.
In order to run this test, you need to install VertexAI SDK
pip install google-cloud-aiplatform>=1.35.0

Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
"""
from langchain_community.embeddings import VertexAIEmbeddings


def test_embedding_documents() -> None:
    documents = ["foo bar"]
    model = VertexAIEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768
    assert model.model_name == model.client._model_id


def test_embedding_query() -> None:
    document = "foo bar"
    model = VertexAIEmbeddings()
    output = model.embed_query(document)
    assert len(output) == 768


def test_large_batches() -> None:
    documents = ["foo bar" for _ in range(0, 251)]
    model_uscentral1 = VertexAIEmbeddings(location="us-central1")
    model_asianortheast1 = VertexAIEmbeddings(location="asia-northeast1")
    model_uscentral1.embed_documents(documents)
    model_asianortheast1.embed_documents(documents)
    assert model_uscentral1.instance["batch_size"] >= 250
    assert model_asianortheast1.instance["batch_size"] < 50


def test_split_by_punctuation() -> None:
    parts = VertexAIEmbeddings._split_by_punctuation("Hello, my friend!\nHow are you?")
    assert parts == [
        "Hello",
        ",",
        " ",
        "my",
        " ",
        "friend",
        "!",
        "\n",
        "How",
        " ",
        "are",
        " ",
        "you",
        "?",
        " ",
        "I",
        " ",
        "have",
        " ",
        "2",
        " ",
        "news",
        ":",
        "\n",
        "\n",
        "-",
        " ",
        "good",
        "\n",
        "-",
        "bad",
    ]


def test_batching() -> None:
    long_text = "foo " * 500  # 1000 words, 2000 tokens
    long_texts = [long_text for _ in range(0, 250)]
    short_text = "foo bar"
    short_texts = [short_text for _ in range(0, 250)]
    model_uscentral1 = VertexAIEmbeddings(location="us-central1")
    five_elem = model_uscentral1._prepare_batches(long_texts, 5)
    # Default batch size is 250
    default250_elem = model_uscentral1._prepare_batches(long_texts)
    two_h_elem = model_uscentral1._prepare_batches(short_texts, 200)
    assert len(five_elem) == 50  # 250/5 items
    assert len(five_elem[0]) == 5  # 5 items per batch
    assert len(default250_elem[0]) == 10  # Should not be more than 20K tokens
    assert len(default250_elem) == 25
    assert len(two_h_elem[0]) == 200  # Short texts can make big batches


def test_paginated_texts() -> None:
    documents = [
        "foo bar",
        "foo baz",
        "bar foo",
        "baz foo",
        "bar bar",
        "foo foo",
        "baz baz",
        "baz bar",
    ]
    model = VertexAIEmbeddings()
    output = model.embed_documents(documents)
    assert len(output) == 8
    assert len(output[0]) == 768
    assert model.model_name == model.client._model_id
