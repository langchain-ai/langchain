"""Test GoogleVertexAI Chat API wrapper."""
import pytest

from langchain_google_vertexai import VertexAI


@pytest.mark.parametrize(
    "model_name",
    [None, "text-bison@001", "gemini-pro"],
)
def test_vertex_initialization(model_name: str) -> None:
    llm = (
        VertexAI(model_name=model_name, project="fake")
        if model_name
        else VertexAI(project="fake")
    )
    assert llm._llm_type == "vertexai"
    try:
        assert llm.model_name == llm.client._model_id
    except AttributeError:
        assert llm.model_name == llm.client._model_name.split("/")[-1]
