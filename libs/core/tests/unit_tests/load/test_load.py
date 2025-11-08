"""Test for Serializable base class."""

from langchain_core.load.load import load


def test_load_with_string_secrets() -> None:
    obj = {"api_key": "__SECRET_API_KEY__"}
    secrets_map = {"__SECRET_API_KEY__": "hello"}
    result = load(obj, secrets_map=secrets_map)

    assert result["api_key"] == "hello"
