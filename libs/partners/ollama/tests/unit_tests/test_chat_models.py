"""Test chat model integration."""

import json

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import ChatOllama, _parse_arguments_from_tool_call


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3-groq-tool-use"}


def test__parse_arguments_from_tool_call() -> None:
    raw_response = '{"model":"sample-model","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_profile_details","arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'  # noqa: E501
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)
