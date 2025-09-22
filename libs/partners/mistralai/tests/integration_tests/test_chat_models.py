"""Test ChatMistral chat model."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import pytest
from httpx import ReadTimeout
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_mistralai.chat_models import ChatMistralAI


def test_stream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    for token in llm.stream("Hello"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in llm.astream("Hello"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or response_metadata. "
            "AIMessageChunk aggregation adds / appends counts and metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert isinstance(full.response_metadata["model_name"], str)
    assert full.response_metadata["model_name"]


async def test_abatch() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert "model_name" in result.response_metadata


def test_invoke() -> None:
    """Test invoke tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_chat_mistralai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatMistralAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


def test_chat_mistralai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatMistralAI(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


def test_chat_mistralai_llm_output_contains_token_usage() -> None:
    """Test llm_output contains model_name."""
    chat = ChatMistralAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert "token_usage" in llm_result.llm_output
    token_usage = llm_result.llm_output["token_usage"]
    assert "prompt_tokens" in token_usage
    assert "completion_tokens" in token_usage
    assert "total_tokens" in token_usage


def test_chat_mistralai_streaming_llm_output_not_contain_token_usage() -> None:
    """Mistral currently doesn't return token usage when streaming."""
    chat = ChatMistralAI(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert "token_usage" in llm_result.llm_output
    token_usage = llm_result.llm_output["token_usage"]
    assert not token_usage


def test_structured_output() -> None:
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)  # type: ignore[call-arg]
    schema = {
        "title": "AnswerWithJustification",
        "description": (
            "An answer to the user question along with justification for the answer."
        ),
        "type": "object",
        "properties": {
            "answer": {"title": "Answer", "type": "string"},
            "justification": {"title": "Justification", "type": "string"},
        },
        "required": ["answer", "justification"],
    }
    structured_llm = llm.with_structured_output(schema)
    result = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(result, dict)


def test_streaming_structured_output() -> None:
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)  # type: ignore[call-arg]

    class Person(BaseModel):
        name: str
        age: int

    structured_llm = llm.with_structured_output(Person)
    strm = structured_llm.stream("Erick, 27 years old")
    for chunk_num, chunk in enumerate(strm):
        assert chunk_num == 0, "should only have one chunk with model"
        assert isinstance(chunk, Person)
        assert chunk.name == "Erick"
        assert chunk.age == 27


class Book(BaseModel):
    name: str
    authors: list[str]


class BookDict(TypedDict):
    name: str
    authors: list[str]


def _check_parsed_result(result: Any, schema: Any) -> None:
    if schema == Book:
        assert isinstance(result, Book)
    else:
        assert all(key in ["name", "authors"] for key in result)


@pytest.mark.parametrize("schema", [Book, BookDict, Book.model_json_schema()])
def test_structured_output_json_schema(schema: Any) -> None:
    llm = ChatMistralAI(model="ministral-8b-latest")  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    messages = [
        {"role": "system", "content": "Extract the book's information."},
        {
            "role": "user",
            "content": "I recently read 'To Kill a Mockingbird' by Harper Lee.",
        },
    ]
    # Test invoke
    result = structured_llm.invoke(messages)
    _check_parsed_result(result, schema)

    # Test stream
    for chunk in structured_llm.stream(messages):
        _check_parsed_result(chunk, schema)


@pytest.mark.parametrize("schema", [Book, BookDict, Book.model_json_schema()])
async def test_structured_output_json_schema_async(schema: Any) -> None:
    llm = ChatMistralAI(model="ministral-8b-latest")  # type: ignore[call-arg]
    structured_llm = llm.with_structured_output(schema, method="json_schema")

    messages = [
        {"role": "system", "content": "Extract the book's information."},
        {
            "role": "user",
            "content": "I recently read 'To Kill a Mockingbird' by Harper Lee.",
        },
    ]
    # Test invoke
    result = await structured_llm.ainvoke(messages)
    _check_parsed_result(result, schema)

    # Test stream
    async for chunk in structured_llm.astream(messages):
        _check_parsed_result(chunk, schema)


def test_tool_call() -> None:
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)  # type: ignore[call-arg]

    class Person(BaseModel):
        name: str
        age: int

    tool_llm = llm.bind_tools([Person])

    result = tool_llm.invoke("Erick, 27 years old")
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call["name"] == "Person"
    assert tool_call["args"] == {"name": "Erick", "age": 27}


def test_streaming_tool_call() -> None:
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)  # type: ignore[call-arg]

    class Person(BaseModel):
        name: str
        age: int

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    strm = tool_llm.stream("Erick, 27 years old")

    additional_kwargs = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }

    assert isinstance(chunk, AIMessageChunk)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "Person"
    assert tool_call_chunk["args"] == '{"name": "Erick", "age": 27}'

    # where it doesn't call the tool
    strm = tool_llm.stream("What is 2+2?")
    acc: Any = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        acc = chunk if acc is None else acc + chunk
    assert acc.content != ""
    assert "tool_calls" not in acc.additional_kwargs


def test_retry_parameters(caplog: pytest.LogCaptureFixture) -> None:
    """Test that retry parameters are honored in ChatMistralAI."""
    # Create a model with intentionally short timeout and multiple retries
    mistral = ChatMistralAI(
        timeout=1,  # Very short timeout to trigger timeouts
        max_retries=3,  # Should retry 3 times
    )

    # Simple test input that should take longer than 1 second to process
    test_input = "Write a 2 sentence story about a cat"

    # Measure start time
    t0 = time.time()

    try:
        # Try to get a response
        response = mistral.invoke(test_input)

        # If successful, validate the response
        elapsed_time = time.time() - t0
        logging.info(f"Request succeeded in {elapsed_time:.2f} seconds")
        # Check that we got a valid response
        assert response.content
        assert isinstance(response.content, str)
        assert "cat" in response.content.lower()

    except ReadTimeout:
        elapsed_time = time.time() - t0
        logging.info(f"Request timed out after {elapsed_time:.2f} seconds")
        assert elapsed_time >= 3.0
        pytest.skip("Test timed out as expected with short timeout")
    except Exception as e:
        logging.error(f"Unexpected exception: {e}")
        raise
