from unittest.mock import patch

import pytest

from langchain_google_vertexai.llms import VertexAI


@pytest.mark.parametrize(
    "params",
    [
        {
            "temperature": 0.8,
            "max_output_tokens": 128,
            "candidate_count": 1,
            "top_k": 30,
            "top_p": 0.9,
        },
        {
            "temperature": 0.0,
            "max_output_tokens": 0,
            "candidate_count": 1,
            "top_k": 0,
            "top_p": 0.0,
        },
    ],
)
def test_set_params_palm_model(params):
    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as _:
        model = VertexAI(model_name="text-bison", **params)
        assert model._default_params == params


@pytest.mark.parametrize(
    "params",
    [
        {
            "temperature": 0.8,
            "max_output_tokens": 128,
            "candidate_count": 1,
        },
        {
            "temperature": 0.0,
            "max_output_tokens": 0,
            "candidate_count": 1,
        },
    ],
)
def test_set_params_codey_model(params):
    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as _:
        model = VertexAI(model_name="code-bison", **params)
        assert model._default_params == params


@pytest.mark.parametrize(
    "params",
    [
        {
            "temperature": 0.8,
            "max_output_tokens": 128,
            "candidate_count": 1,
            "top_k": 30,
            "top_p": 0.9,
        },
        {
            "temperature": 0.0,
            "max_output_tokens": 0,
            "candidate_count": 1,
            "top_k": 0,
            "top_p": 0.0,
        },
    ],
)
def test_set_params_gemini_model(params):
    with patch("langchain_google_vertexai.llms.GenerativeModel") as _:
        model = VertexAI(model_name="gemini-pro", **params)

        assert model._default_params == params
