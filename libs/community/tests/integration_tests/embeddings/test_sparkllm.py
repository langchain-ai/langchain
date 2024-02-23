"""Test SparkLLM Text Embedding."""
from langchain_community.embeddings.sparkllm import SparkLLMTextEmbeddings


def test_baichuan_embedding_documents() -> None:
    """Test SparkLLM Text Embedding for documents."""
    documents = [
        "iFLYTEK is a well-known intelligent speech and artificial intelligence "
        "publicly listed company in the Asia-Pacific Region. Since its establishment,"
        "the company is devoted to cornerstone technological research "
        "in speech and languages, natural language understanding, machine learning,"
        "machine reasoning, adaptive learning, "
        "and has maintained the world-leading position in those "
        "domains. The company actively promotes the development of A.I. "
        "products and their sector-based "
        "applications, with visions of enabling machines to listen and speak, "
        "understand and think, "
        "creating a better world with artificial intelligence."
    ]
    embedding = SparkLLMTextEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1  # type: ignore[arg-type]
    assert len(output[0]) == 2560  # type: ignore[index]


def test_baichuan_embedding_query() -> None:
    """Test SparkLLM Text Embedding for query."""
    document = (
        "iFLYTEK Open Platform was launched in 2010 by iFLYTEK as China’s "
        "first Artificial Intelligence open platform for Mobile Internet "
        "and intelligent hardware developers"
    )
    embedding = SparkLLMTextEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 2560  # type: ignore[arg-type]
