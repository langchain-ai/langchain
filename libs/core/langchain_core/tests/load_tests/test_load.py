import pytest
from langchain_core.load.load import load


@pytest.mark.requires("openai", "langchain_openai")
def test_load_secrets_map_on_string_values():
    obj = {"api_key": "__MY_SECRET__"}
    secrets_map = {"__MY_SECRET__": "super_secret_token"}
    result = load(obj, secrets_map=secrets_map)
    assert result["api_key"] == "super_secret_token"
