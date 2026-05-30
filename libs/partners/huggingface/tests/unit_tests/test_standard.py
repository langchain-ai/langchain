"""Standard LangChain interface tests."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class TestHuggingFaceChatModelUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatHuggingFace

    @property
    def chat_model_params(self) -> dict:
        llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
            repo_id="meta-llama/Llama-3.3-70B-Instruct",
            task="conversational",
            provider="together",
            temperature=0,
        )
        return {"llm": llm}

    @pytest.fixture
    def model(self, request: Any) -> BaseChatModel:
        return self.chat_model_class(**self.chat_model_params)  # type: ignore[call-arg]
