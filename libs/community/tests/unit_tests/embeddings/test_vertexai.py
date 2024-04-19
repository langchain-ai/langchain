"""Test Vertex AI embeddings API wrapper."""

from langchain_community.embeddings import VertexAIEmbeddings


def test_split_by_punctuation() -> None:
    parts = VertexAIEmbeddings._split_by_punctuation(
        "Hello, my friend!\nHow are you?\nI have 2 news:\n\n\t- Good,\n\t- Bad."
    )
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
        "\n",
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
        "\t",
        "-",
        " ",
        "Good",
        ",",
        "\n",
        "\t",
        "-",
        " ",
        "Bad",
        ".",
    ]


def test_batching() -> None:
    long_text = "foo " * 500  # 1000 words, 2000 tokens
    long_texts = [long_text for _ in range(0, 250)]
    documents251 = ["foo bar" for _ in range(0, 251)]
    five_elem = VertexAIEmbeddings._prepare_batches(long_texts, 5)
    default250_elem = VertexAIEmbeddings._prepare_batches(long_texts, 250)
    batches251 = VertexAIEmbeddings._prepare_batches(documents251, 250)
    assert len(five_elem) == 50  # 250/5 items
    assert len(five_elem[0]) == 5  # 5 items per batch
    assert len(default250_elem[0]) == 10  # Should not be more than 20K tokens
    assert len(default250_elem) == 25
    assert len(batches251[0]) == 250
    assert len(batches251[1]) == 1
