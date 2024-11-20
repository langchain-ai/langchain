"""Ollama specific chat model integration tests"""

from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


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
