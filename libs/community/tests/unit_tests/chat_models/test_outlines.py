import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import BaseModel, Field

from langchain_community.chat_models.outlines import ChatOutlines


def test_chat_outlines_initialization(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)

    chat = ChatOutlines(
        model="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=42,
        stop=["\n"],
    )
    assert chat.model == "microsoft/Phi-3-mini-4k-instruct"
    assert chat.max_tokens == 42
    assert chat.backend == "transformers"
    assert chat.stop == ["\n"]


def test_chat_outlines_backend_llamacpp(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)
    chat = ChatOutlines(
        model="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
        backend="llamacpp",
    )
    assert chat.backend == "llamacpp"


def test_chat_outlines_backend_vllm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", backend="vllm")
    assert chat.backend == "vllm"


def test_chat_outlines_backend_mlxlm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", backend="mlxlm")
    assert chat.backend == "mlxlm"


def test_chat_outlines_with_regex(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)
    regex = r"\d{3}-\d{3}-\d{4}"
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", regex=regex)
    assert chat.regex == regex


def test_chat_outlines_with_type_constraints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", type_constraints=int)
    assert chat.type_constraints == int  # noqa


def test_chat_outlines_with_json_schema(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)

    class TestSchema(BaseModel):
        name: str = Field(description="A person's name")
        age: int = Field(description="A person's age")

    chat = ChatOutlines(
        model="microsoft/Phi-3-mini-4k-instruct", json_schema=TestSchema
    )
    assert chat.json_schema == TestSchema


def test_chat_outlines_with_grammar(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)

    grammar = """
?start: expression
?expression: term (("+" | "-") term)*
?term: factor (("*" | "/") factor)*
?factor: NUMBER | "-" factor | "(" expression ")"
%import common.NUMBER
    """
    chat = ChatOutlines(model="microsoft/Phi-3-mini-4k-instruct", grammar=grammar)
    assert chat.grammar == grammar


def test_raise_for_multiple_output_constraints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(ChatOutlines, "build_client", lambda self: self)

    with pytest.raises(ValueError):
        ChatOutlines(
            model="microsoft/Phi-3-mini-4k-instruct",
            type_constraints=int,
            regex=r"\d{3}-\d{3}-\d{4}",
        )
