"""Test AI Mind Chat wrapper."""

import pytest
from typing import List

from langchain_community.chat_models import ChatAIMind


@pytest.mark.requires("openai")
def test_chat_ai_mind_model_params() -> None:
    test_cases: List[dict] = [
        {"model_name": "foo", "mindsdb_api_key": "foo"},
        {"model": "foo", "mindsdb_api_key": "foo"},
        {"model_name": "foo", "mindsdb_api_key": "foo"},
        {"model_name": "foo", "mindsdb_api_key": "foo", "max_retries": 2},
    ]

    for case in test_cases:
        llm = ChatAIMind(**case)
        assert llm.model_name == "foo", "Model name should be 'foo'"
        assert llm.mindsdb_api_key == "foo", "API key should be 'foo'"
        assert hasattr(llm, "max_retries"), "max_retries attribute should exist"
        assert llm.max_retries == 2, "max_retries default should be set to 2"