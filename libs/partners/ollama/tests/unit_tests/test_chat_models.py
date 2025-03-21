"""Test chat model integration."""

from contextlib import contextmanager
from typing import Any, Dict, Type

from langchain_tests.unit_tests import ChatModelUnitTests

from httpx import Request, Response
import httpx
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import ChatMessage

import pytest
class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> Dict:
        return {"model": "llama3-groq-tool-use"}

@contextmanager
def _mock_httpx_client_stream(*args: Any, **kwargs: Any):
  yield Response(status_code=200, content='{"message": {"role":"assistant","content":"The meaning of life is..."}}', request=Request(method="POST", url="http://whocares:11434")) 

def test_arbitrary_roles_accepted_in_chatmessages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(httpx.Client, 'stream', _mock_httpx_client_stream)

    llm = ChatOllama(
        base_url    = "http://whocares:11434",
        model       = "granite3.2",
        verbose     = True,
    )

    messages = [
                    ChatMessage(role="somerandomrole", content="I'm ok with you adding any role message now!"),
                    ChatMessage(role="control", content="thinking"),
                    ChatMessage(role="user",content="What is the meaning of life?")
                ]

    llm.invoke(messages)
