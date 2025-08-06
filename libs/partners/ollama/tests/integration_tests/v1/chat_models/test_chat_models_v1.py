"""Ollama-specific v1 chat model integration tests.

Standard tests are handled in `test_chat_models_v1_standard.py`.

"""

from __future__ import annotations

from typing import Annotated, Optional

import pytest
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_ollama.v1.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"


@pytest.mark.parametrize(("method"), [("function_calling"), ("json_schema")])
def test_structured_output(method: str) -> None:
    """Test to verify structured output via tool calling and `format` parameter."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)
    query = "Tell me a joke about cats."

    # Pydantic
    if method == "function_calling":
        structured_llm = llm.with_structured_output(Joke, method="function_calling")
        result = structured_llm.invoke(query)
        assert isinstance(result, Joke)

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, Joke)

    # JSON Schema
    if method == "json_schema":
        structured_llm = llm.with_structured_output(
            Joke.model_json_schema(), method="json_schema"
        )
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}

        # Typed Dict
        class JokeSchema(TypedDict):
            """Joke to tell user."""

            setup: Annotated[str, "question to set up a joke"]
            punchline: Annotated[str, "answer to resolve the joke"]

        structured_llm = llm.with_structured_output(JokeSchema, method="json_schema")
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}


@pytest.mark.parametrize(("model"), [(DEFAULT_MODEL_NAME)])
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

        people: list[Person]

    chat = llm.with_structured_output(Data)
    text = (
        "Alan Smith is 6 feet tall and has blond hair."
        "Alan Poe is 3 feet tall and has grey hair."
    )
    result = chat.invoke(text)
    assert isinstance(result, Data)

    for chunk in chat.stream(text):
        assert isinstance(chunk, Data)


# def test_reasoning_content_blocks() -> None:
#     """Test that the model supports reasoning content blocks."""
#     llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)

#     # Test with a reasoning prompt
#     messages = [HumanMessage("Think step by step and solve: What is 2 + 2?")]

#     result = llm.invoke(messages)

#     # Check that we get an AIMessage with content blocks
#     assert isinstance(result, AIMessage)
#     assert len(result.content) > 0

#     # For streaming, check that reasoning blocks are properly handled
#     chunks = []
#     for chunk in llm.stream(messages):
#         chunks.append(chunk)
#         assert isinstance(chunk, AIMessageChunk)

#     assert len(chunks) > 0


# def test_multimodal_support() -> None:
#     """Test that the model supports image content blocks."""
#     llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)

#     # Create a message with image content block
#     from langchain_core.messages.content_blocks import (
#         create_image_block,
#         create_text_block,
#     )

#     # Test with a simple base64 placeholder (real integration would use actual image)
#     message = HumanMessage(
#         content=[
#             create_text_block("Describe this image:"),
#             create_image_block(
#                 base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501
#             ),
#         ]
#     )

#     result = llm.invoke([message])

#     # Check that we get a response (even if it's just acknowledging the image)
#     assert isinstance(result, AIMessage)
#     assert len(result.content) > 0
