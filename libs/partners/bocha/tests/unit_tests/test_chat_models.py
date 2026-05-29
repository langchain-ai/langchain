import os
from unittest.mock import patch
from langchain_bocha import ChatBocha


def test_chat_bocha_init():
    with patch.dict("os.environ", {"BOCHA_API_KEY": "test-key"}):
        llm = ChatBocha(model="deepseek-v4-pro")
        assert llm.model_name == "deepseek-v4-pro"
        assert "bocha.ai" in llm.openai_api_base


def test_chat_bocha_default_model():
    with patch.dict("os.environ", {"BOCHA_API_KEY": "test-key"}):
        llm = ChatBocha()
        assert llm.model_name == "deepseek-v4-pro"


def test_chat_bocha_custom_key():
    llm = ChatBocha(bocha_api_key="my-custom-key")
    assert llm.openai_api_key.get_secret_value() == "my-custom-key"
