import platform

import pytest
from pydantic import BaseModel, Field

from langchain_community.chat_models.outlines import ChatOutlines


def test_chat_outlines_initialization():
    chat = ChatOutlines(
        model="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=42,
        stop=["\n"],
    )
    assert chat.model == "microsoft/Phi-3-mini-4k-instruct"
    assert chat.max_tokens == 42
    assert chat.backend == "transformers"
    assert chat.stop == ["\n"]


def test_chat_outlines_backend_llamacpp():
    chat = ChatOutlines(
        model="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
        backend="llamacpp",
    )
    assert chat.backend == "llamacpp"


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="vLLM backend is not supported on macOS"
)
def test_chat_outlines_backend_vllm():
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", backend="vllm")
    assert chat.backend == "vllm"


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="MLX backend is only supported on macOS"
)
def test_chat_outlines_backend_mlxlm():
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", backend="mlxlm")
    assert chat.backend == "mlxlm"


def test_chat_outlines_with_regex():
    regex = r"\d{3}-\d{3}-\d{4}"
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", regex=regex)
    assert chat.regex == regex


def test_chat_outlines_with_type_constraints():
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", type_constraints=int)
    assert chat.type_constraints == int  # noqa


def test_chat_outlines_with_json_schema():
    class TestSchema(BaseModel):
        name: str = Field(description="A person's name")
        age: int = Field(description="A person's age")

    chat = ChatOutlines(
        model="microsoft/Phi-3-mini-4k-instruct", json_schema=TestSchema
    )
    assert chat.json_schema == TestSchema


def test_chat_outlines_with_grammar():
    grammar = """
    ?start: expression
    ?expression: term (("+" | "-") term)*
    ?term: factor (("*" | "/") factor)*
    ?factor: NUMBER | "-" factor | "(" expression ")"
    %import common.NUMBER
    """
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", grammar=grammar)
    assert chat.grammar == grammar


def test_raise_error_for_invalid_backend():
    with pytest.raises(ValueError, match="Unsupported backend: invalid_backend"):
        ChatOutlines(
            model="microsoft/Phi-3-mini-4k-instruct", backend="invalid_backend"
        )


def test_raise_for_multiple_output_constraints():
    with pytest.raises(ValueError):
        ChatOutlines(
            model="microsoft/Phi-3-mini-4k-instruct",
            type_constraints=int,
            regex=r"\d{3}-\d{3}-\d{4}",
        )
