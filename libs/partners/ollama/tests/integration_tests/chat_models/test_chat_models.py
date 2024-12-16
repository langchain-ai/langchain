"""Ollama specific chat model integration tests"""

from typing import List, Optional

import pytest
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_ollama import ChatOllama


@pytest.mark.parametrize(("method"), [("function_calling"), ("json_schema")])
def test_structured_output(method: str) -> None:
    """Test to verify structured output via tool calling and ``format`` parameter."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm = ChatOllama(model="llama3.1", temperature=0)
    query = "Tell me a joke about cats."

    # Pydantic
    structured_llm = llm.with_structured_output(Joke, method=method)  # type: ignore[arg-type]
    result = structured_llm.invoke(query)
    assert isinstance(result, Joke)

    for chunk in structured_llm.stream(query):
        assert isinstance(chunk, Joke)

    # JSON Schema
    structured_llm = llm.with_structured_output(Joke.model_json_schema(), method=method)  # type: ignore[arg-type]
    result = structured_llm.invoke(query)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline"}

    for chunk in structured_llm.stream(query):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline"}

    # Typed Dict
    class JokeSchema(TypedDict):
        """Joke to tell user."""

        setup: Annotated[str, "question to set up a joke"]
        punchline: Annotated[str, "answer to resolve the joke"]

    structured_llm = llm.with_structured_output(JokeSchema, method=method)  # type: ignore[arg-type]
    result = structured_llm.invoke(query)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline"}

    for chunk in structured_llm.stream(query):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline"}


@pytest.mark.parametrize(("model"), [("llama3.1")])
def test_structured_output_deeply_nested(model: str) -> None:
    """Test to verify structured output with a nested objects."""
    llm = ChatOllama(model=model, temperature=0)

    class Person(BaseModel):
        """Information about a person."""

        name: Optional[str] = Field(default=None, description="The name of the person")
        hair_color: Optional[str] = Field(
            default=None, description="The color of the person's hair if known"
        )
        height_in_meters: Optional[str] = Field(
            default=None, description="Height measured in meters"
        )

    class Data(BaseModel):
        """Extracted data about people."""

        people: List[Person]

    chat = llm.with_structured_output(Data)  # type: ignore[arg-type]
    text = (
        "Alan Smith is 6 feet tall and has blond hair."
        "Alan Poe is 3 feet tall and has grey hair."
    )
    result = chat.invoke(text)
    assert isinstance(result, Data)

    for chunk in chat.stream(text):
        assert isinstance(chunk, Data)
