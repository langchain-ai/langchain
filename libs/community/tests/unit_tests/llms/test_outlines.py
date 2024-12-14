import pytest
from _pytest.monkeypatch import MonkeyPatch

from langchain_community.llms.outlines import Outlines


def test_outlines_initialization(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)

    llm = Outlines(
        model="microsoft/Phi-3-mini-4k-instruct",
        max_tokens=42,
        stop=["\n"],
    )
    assert llm.model == "microsoft/Phi-3-mini-4k-instruct"
    assert llm.max_tokens == 42
    assert llm.backend == "transformers"
    assert llm.stop == ["\n"]


def test_outlines_backend_llamacpp(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    llm = Outlines(
        model="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
        backend="llamacpp",
    )
    assert llm.backend == "llamacpp"


def test_outlines_backend_vllm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", backend="vllm")
    assert llm.backend == "vllm"


def test_outlines_backend_mlxlm(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", backend="mlxlm")
    assert llm.backend == "mlxlm"


def test_outlines_with_regex(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    regex = r"\d{3}-\d{3}-\d{4}"
    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", regex=regex)
    assert llm.regex == regex


def test_outlines_with_type_constraints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", type_constraints=int)
    assert llm.type_constraints == int  # noqa


def test_outlines_with_json_schema(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    from pydantic import BaseModel, Field

    class TestSchema(BaseModel):
        name: str = Field(description="A person's name")
        age: int = Field(description="A person's age")

    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", json_schema=TestSchema)
    assert llm.json_schema == TestSchema


def test_outlines_with_grammar(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    grammar = """
    ?start: expression
    ?expression: term (("+" | "-") term)*
    ?term: factor (("*" | "/") factor)*
    ?factor: NUMBER | "-" factor | "(" expression ")"
    %import common.NUMBER
    """
    llm = Outlines(model="microsoft/Phi-3-mini-4k-instruct", grammar=grammar)
    assert llm.grammar == grammar


def test_raise_for_multiple_output_constraints(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Outlines, "build_client", lambda self: self)
    with pytest.raises(ValueError):
        Outlines(
            model="microsoft/Phi-3-mini-4k-instruct",
            type_constraints=int,
            regex=r"\d{3}-\d{3}-\d{4}",
        )
        Outlines(
            model="microsoft/Phi-3-mini-4k-instruct",
            type_constraints=int,
            regex=r"\d{3}-\d{3}-\d{4}",
        )
