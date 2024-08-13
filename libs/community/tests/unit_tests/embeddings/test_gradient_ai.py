import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.embeddings import GradientEmbeddings

_MODEL_ID = "my_model_valid_id"
_GRADIENT_SECRET = "secret_valid_token_123456"
_GRADIENT_WORKSPACE_ID = "valid_workspace_12345"
_GRADIENT_BASE_URL = "https://api.gradient.ai/api"
_DOCUMENTS = [
    "pizza",
    "another long pizza",
    "a document",
    "another long pizza",
    "super long document with many tokens",
]


class GradientEmbeddingsModel(MagicMock):
    """MockGradientModel."""

    def embed(self, inputs: List[Dict[str, str]]) -> Any:
        """Just duplicate the query m times."""
        output = MagicMock()

        embeddings = []
        for i, inp in enumerate(inputs):
            # verify correct ordering
            inp = inp["input"]  # type: ignore[assignment]
            if "pizza" in inp:
                v = [1.0, 0.0, 0.0]
            elif "document" in inp:
                v = [0.0, 0.9, 0.0]
            else:
                v = [0.0, 0.0, -1.0]
            if len(inp) > 10:
                v[2] += 0.1
            output_inner = MagicMock()
            output_inner.embedding = v
            embeddings.append(output_inner)

        output.embeddings = embeddings
        return output

    async def aembed(self, *args) -> Any:  # type: ignore[no-untyped-def]
        return self.embed(*args)


class MockGradient(MagicMock):
    """Mock Gradient package."""

    def __init__(self, access_token: str, workspace_id, host):  # type: ignore[no-untyped-def]
        assert access_token == _GRADIENT_SECRET
        assert workspace_id == _GRADIENT_WORKSPACE_ID
        assert host == _GRADIENT_BASE_URL

    def get_embeddings_model(self, slug: str) -> GradientEmbeddingsModel:
        assert slug == _MODEL_ID
        return GradientEmbeddingsModel()

    def close(self) -> None:
        """Mock Gradient close."""
        return


class MockGradientaiPackage(MagicMock):
    """Mock Gradientai package."""

    Gradient = MockGradient
    __version__: str = "1.4.0"


def test_gradient_llm_sync() -> None:
    with patch.dict(sys.modules, {"gradientai": MockGradientaiPackage()}):
        embedder = GradientEmbeddings(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )
        assert embedder.gradient_access_token == _GRADIENT_SECRET
        assert embedder.gradient_api_url == _GRADIENT_BASE_URL
        assert embedder.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
        assert embedder.model == _MODEL_ID

        response = embedder.embed_documents(_DOCUMENTS)
        want = [
            [1.0, 0.0, 0.0],  # pizza
            [1.0, 0.0, 0.1],  # pizza  + long
            [0.0, 0.9, 0.0],  # doc
            [1.0, 0.0, 0.1],  # pizza + long
            [0.0, 0.9, 0.1],  # doc + long
        ]

        assert response == want


def test_gradient_wrong_setup() -> None:
    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token="",  # empty
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )


def test_gradient_wrong_setup2() -> None:
    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id="",  # empty
            model=_MODEL_ID,
        )


def test_gradient_wrong_setup3() -> None:
    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url="-",  # empty
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )
