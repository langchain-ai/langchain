import os

import pytest

from langchain.llms.anyscale import Anyscale

os.environ["ANYSCALE_API_KEY"] = SECRET_ANYSCALE_API_KEY = "SECRET_ANYSCALE_API_KEY"


@pytest.mark.requires("openai")
def test_initialization_with_api_key_as_env_var() -> None:
    """Test initialization with an API key provided via an env variable"""
    llm = Anyscale(
        anyscale_api_base="test",
        model_name="test",
    )
    _string = str(llm)
    _repr = repr(llm)

    assert SECRET_ANYSCALE_API_KEY not in _string
    assert SECRET_ANYSCALE_API_KEY not in _repr


@pytest.mark.requires("openai")
def test_initialization_with_api_key_as_initializer_argument() -> None:
    """Test initialization with an API key provided via the initializer"""
    llm = Anyscale(
        anyscale_api_base="test",
        anyscale_api_key=os.environ["ANYSCALE_API_KEY"],
        model_name="test",
    )
    _string = str(llm)
    _repr = repr(llm)

    assert SECRET_ANYSCALE_API_KEY not in _string
    assert SECRET_ANYSCALE_API_KEY not in _repr
