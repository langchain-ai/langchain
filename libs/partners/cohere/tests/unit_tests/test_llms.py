"""Test Cohere API wrapper."""


from langchain_core.pydantic_v1 import SecretStr
from pytest import MonkeyPatch

from langchain_cohere.llms import BaseCohere


def test_cohere_api_key(monkeypatch: MonkeyPatch) -> None:
    """Test that cohere api key is a secret key."""
    # test initialization from init
    assert isinstance(BaseCohere(cohere_api_key="1").cohere_api_key, SecretStr)

    # test initialization from env variable
    monkeypatch.setenv("COHERE_API_KEY", "secret-api-key")
    assert isinstance(BaseCohere().cohere_api_key, SecretStr)


# def test_saving_loading_llm(tmp_path: Path) -> None:
#     """Test saving/loading an Cohere LLM."""
#     llm = BaseCohere(max_tokens=10)
#     llm.save(file_path=tmp_path / "cohere.yaml")
#     loaded_llm = load_llm(tmp_path / "cohere.yaml")
#     assert_llm_equality(llm, loaded_llm)
