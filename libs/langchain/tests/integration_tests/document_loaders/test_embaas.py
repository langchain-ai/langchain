from typing import Any
from unittest.mock import MagicMock, patch

import responses

from langchain.document_loaders import EmbaasBlobLoader, EmbaasLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.embaas import EMBAAS_DOC_API_URL


@responses.activate
def test_handle_request() -> None:
    responses.add(
        responses.POST,
        EMBAAS_DOC_API_URL,
        json={
            "data": {
                "chunks": [
                    {
                        "text": "Hello",
                        "metadata": {"start_page": 1, "end_page": 2},
                        "embeddings": [0.0],
                    }
                ]
            }
        },
        status=200,
    )

    loader = EmbaasBlobLoader(embaas_api_key="api_key", params={"should_embed": True})
    documents = loader.parse(blob=Blob.from_data(data="Hello"))
    assert len(documents) == 1
    assert documents[0].page_content == "Hello"
    assert documents[0].metadata["start_page"] == 1
    assert documents[0].metadata["end_page"] == 2
    assert documents[0].metadata["embeddings"] == [0.0]


@responses.activate
def test_handle_request_exception() -> None:
    responses.add(
        responses.POST,
        EMBAAS_DOC_API_URL,
        json={"message": "Invalid request"},
        status=400,
    )
    loader = EmbaasBlobLoader(embaas_api_key="api_key")
    try:
        loader.parse(blob=Blob.from_data(data="Hello"))
    except Exception as e:
        assert "Invalid request" in str(e)


@patch.object(EmbaasBlobLoader, "_handle_request")
def test_load(mock_handle_request: Any) -> None:
    mock_handle_request.return_value = [MagicMock()]
    loader = EmbaasLoader(file_path="test_embaas.py", embaas_api_key="api_key")
    documents = loader.load()
    assert len(documents) == 1
