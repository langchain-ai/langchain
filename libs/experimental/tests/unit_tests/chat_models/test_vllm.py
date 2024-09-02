"""Test vLLM Chat wrapper."""

from importlib import import_module

import pytest
from langchain_community.chat_models.vllm import ChatVLLMOpenAI
from langchain_core.pydantic_v1 import BaseModel


class TestChatVLLMOpenAI:
    def test_import_class(self) -> None:
        """Test that the class can be imported."""
        module_name = "langchain_community.chat_models.vllm"
        class_name = "ChatVLLMOpenAI"

        module = import_module(module_name)
        assert hasattr(module, class_name)

    def test_default_openai_api_base(self) -> None:
        chat = ChatVLLMOpenAI()
        assert chat.openai_api_base == "http://localhost:8000/v1"

    def test_default_params(self) -> None:
        chat = ChatVLLMOpenAI()
        assert "extra_body" in chat._default_params

    def test_extra_body(self) -> None:
        chat = ChatVLLMOpenAI(extra_body={"test": "test"})
        assert chat.extra_body == {"test": "test"}

    def test_llm_type(self) -> None:
        chat = ChatVLLMOpenAI()
        assert chat._llm_type == "vllm-openai"

    def test_get_instructions(self) -> None:
        chat = ChatVLLMOpenAI()
        for mode in ["guided_json", "guided_regex", "guided_choice", "guided_grammar"]:
            assert isinstance(chat._get_instructions("schema", mode), str)
        with pytest.raises(ValueError):
            chat._get_instructions("schema", "unsupported_mode")

    def test_with_structured_output(self) -> None:
        class TestSchema(BaseModel):
            test: str

        chat = ChatVLLMOpenAI()
        with pytest.raises(ValueError):
            chat.with_structured_output(schema=123)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            chat.with_structured_output(schema=TestSchema, unsupported_arg=True)

        chat.with_structured_output(schema=TestSchema)
