import numpy as np
import pytest

from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

_MODEL = "models/embedding-001"


@pytest.mark.parametrize(
    "query",
    [
        "Hi",
        "This is a longer query string to test the embedding functionality of the"
        " model against the pickle rick?",
    ],
)
def test_embed_query_different_lengths(query: str) -> None:
    """Test embedding queries of different lengths."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL)
    result = model.embed_query(query)
    assert len(result) == 768


@pytest.mark.parametrize(
    "query",
    [
        "Hi",
        "This is a longer query string to test the embedding functionality of the"
        " model against the pickle rick?",
    ],
)
async def test_aembed_query_different_lengths(query: str) -> None:
    """Test embedding queries of different lengths."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL)
    result = await model.aembed_query(query)
    assert len(result) == 768


def test_embed_documents() -> None:
    """Test embedding a query."""
    model = GoogleGenerativeAIEmbeddings(
        model=_MODEL,
    )
    result = model.embed_documents(["Hello world", "Good day, world"])
    assert len(result) == 2
    assert len(result[0]) == 768
    assert len(result[1]) == 768


async def test_aembed_documents() -> None:
    """Test embedding a query."""
    model = GoogleGenerativeAIEmbeddings(
        model=_MODEL,
    )
    result = await model.aembed_documents(["Hello world", "Good day, world"])
    assert len(result) == 2
    assert len(result[0]) == 768
    assert len(result[1]) == 768


def test_invalid_model_error_handling() -> None:
    """Test error handling with an invalid model name."""
    with pytest.raises(GoogleGenerativeAIError):
        GoogleGenerativeAIEmbeddings(model="invalid_model").embed_query("Hello world")


def test_invalid_api_key_error_handling() -> None:
    """Test error handling with an invalid API key."""
    with pytest.raises(GoogleGenerativeAIError):
        GoogleGenerativeAIEmbeddings(
            model=_MODEL, google_api_key="invalid_key"
        ).embed_query("Hello world")


def test_embed_documents_consistency() -> None:
    """Test embedding consistency for the same document."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL)
    doc = "Consistent document for testing"
    result1 = model.embed_documents([doc])
    result2 = model.embed_documents([doc])
    assert result1 == result2


def test_embed_documents_quality() -> None:
    """Smoke test embedding quality by comparing similar and dissimilar documents."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL)
    similar_docs = ["Document A", "Similar Document A"]
    dissimilar_docs = ["Document A", "Completely Different Zebra"]
    similar_embeddings = model.embed_documents(similar_docs)
    dissimilar_embeddings = model.embed_documents(dissimilar_docs)
    similar_distance = np.linalg.norm(
        np.array(similar_embeddings[0]) - np.array(similar_embeddings[1])
    )
    dissimilar_distance = np.linalg.norm(
        np.array(dissimilar_embeddings[0]) - np.array(dissimilar_embeddings[1])
    )
    assert similar_distance < dissimilar_distance
