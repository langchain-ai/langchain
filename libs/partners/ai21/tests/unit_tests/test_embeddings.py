"""Test embedding model integration."""

from typing import List
from unittest.mock import Mock

import pytest
from ai21 import AI21Client, MissingApiKeyError
from ai21.models import EmbedResponse, EmbedResult, EmbedType
from pytest_mock import MockerFixture

from langchain_ai21.embeddings import AI21Embeddings
from tests.unit_tests.conftest import DUMMY_API_KEY, temporarily_unset_api_key

_EXAMPLE_EMBEDDING_0 = [1.0, 2.0, 3.0]
_EXAMPLE_EMBEDDING_1 = [4.0, 5.0, 6.0]
_EXAMPLE_EMBEDDING_2 = [7.0, 8.0, 9.0]

_EXAMPLE_EMBEDDING_RESPONSE = EmbedResponse(
    results=[
        EmbedResult(_EXAMPLE_EMBEDDING_0),
        EmbedResult(_EXAMPLE_EMBEDDING_1),
        EmbedResult(_EXAMPLE_EMBEDDING_2),
    ],
    id="test_id",
)


def test_initialization__when_no_api_key__should_raise_exception() -> None:
    """Test integration initialization."""
    with temporarily_unset_api_key():
        with pytest.raises(MissingApiKeyError):
            AI21Embeddings()


@pytest.fixture
def mock_client_with_embeddings(mocker: MockerFixture) -> Mock:
    mock_client = mocker.MagicMock(spec=AI21Client)
    mock_client.embed = mocker.MagicMock()
    mock_client.embed.create.return_value = _EXAMPLE_EMBEDDING_RESPONSE

    return mock_client


def test_embed_query(mock_client_with_embeddings: Mock) -> None:
    llm = AI21Embeddings(client=mock_client_with_embeddings, api_key=DUMMY_API_KEY)  # type: ignore[arg-type]

    text = "Hello embeddings world!"
    response = llm.embed_query(text=text)
    assert response == _EXAMPLE_EMBEDDING_0
    mock_client_with_embeddings.embed.create.assert_called_once_with(
        texts=[text],
        type=EmbedType.QUERY,
    )


def test_embed_documents(mock_client_with_embeddings: Mock) -> None:
    llm = AI21Embeddings(client=mock_client_with_embeddings, api_key=DUMMY_API_KEY)  # type: ignore[arg-type]

    texts = ["Hello embeddings world!", "Some other text", "Some more text"]
    response = llm.embed_documents(texts=texts)
    assert response == [
        _EXAMPLE_EMBEDDING_0,
        _EXAMPLE_EMBEDDING_1,
        _EXAMPLE_EMBEDDING_2,
    ]
    mock_client_with_embeddings.embed.create.assert_called_once_with(
        texts=texts,
        type=EmbedType.SEGMENT,
    )


@pytest.mark.parametrize(
    ids=[
        "empty_texts",
        "chunk_size_greater_than_texts_length",
        "chunk_size_equal_to_texts_length",
        "chunk_size_less_than_texts_length",
        "chunk_size_one_with_multiple_texts",
        "chunk_size_greater_than_texts_length",
    ],
    argnames=["texts", "chunk_size", "expected_internal_embeddings_calls"],
    argvalues=[
        ([], 3, 0),
        (["text1", "text2", "text3"], 5, 1),
        (["text1", "text2", "text3"], 3, 1),
        (["text1", "text2", "text3", "text4", "text5"], 2, 3),
        (["text1", "text2", "text3"], 1, 3),
        (["text1", "text2", "text3"], 10, 1),
    ],
)
def test_get_len_safe_embeddings(
    mock_client_with_embeddings: Mock,
    texts: List[str],
    chunk_size: int,
    expected_internal_embeddings_calls: int,
) -> None:
    llm = AI21Embeddings(client=mock_client_with_embeddings, api_key=DUMMY_API_KEY)  # type: ignore[arg-type]
    llm.embed_documents(texts=texts, batch_size=chunk_size)
    assert (
        mock_client_with_embeddings.embed.create.call_count
        == expected_internal_embeddings_calls
    )
