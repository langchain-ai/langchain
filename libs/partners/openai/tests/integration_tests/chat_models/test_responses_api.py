"""Test Responses API usage."""

import os
from typing import Any, Optional, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)

from langchain_openai import ChatOpenAI


def _check_response(response: Optional[BaseMessage]) -> None:
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "text":
            assert isinstance(block["text"], str)
            for annotation in block["annotations"]:
                if annotation["type"] == "file_citation":
                    assert all(
                        key in annotation
                        for key in ["file_id", "filename", "index", "type"]
                    )
                elif annotation["type"] == "web_search":
                    assert all(
                        key in annotation
                        for key in ["end_index", "start_index", "title", "type", "url"]
                    )

    text_content = response.text()
    assert isinstance(text_content, str)
    assert text_content
    assert response.usage_metadata
    assert response.usage_metadata["input_tokens"] > 0
    assert response.usage_metadata["output_tokens"] > 0
    assert response.usage_metadata["total_tokens"] > 0
    assert response.response_metadata["model_name"]
    for tool_output in response.additional_kwargs["tool_outputs"]:
        assert tool_output["id"]
        assert tool_output["status"]
        assert tool_output["type"]


def test_web_search() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")
    first_response = llm.invoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(first_response)

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    _check_response(full)

    # Use OpenAI's stateful API
    response = llm.invoke(
        "what about a negative one",
        tools=[{"type": "web_search_preview"}],
        previous_response_id=first_response.response_metadata["id"],
    )
    _check_response(response)

    # Manually pass in chat history
    response = llm.invoke(
        [
            first_response,
            {
                "role": "user",
                "content": [{"type": "text", "text": "what about a negative one"}],
            },
        ],
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)

    # Bind tool
    response = llm.bind_tools([{"type": "web_search_preview"}]).invoke(
        "What was a positive news story from today?"
    )
    _check_response(response)


async def test_web_search_async() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = await llm.ainvoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)
    assert response.response_metadata["status"]

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)


def test_function_calling() -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model="gpt-4o-mini")
    bound_llm = llm.bind_tools([multiply, {"type": "web_search_preview"}])
    ai_msg = cast(AIMessage, bound_llm.invoke("whats 5 * 4"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}

    full: Any = None
    for chunk in bound_llm.stream("whats 5 * 4"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["name"] == "multiply"
    assert set(full.tool_calls[0]["args"]) == {"x", "y"}

    response = bound_llm.invoke("whats some good news from today")
    _check_response(response)


def test_stateful_api() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)
    response = llm.invoke("how are you, my name is Bobo")
    assert "id" in response.response_metadata

    second_response = llm.invoke(
        "what's my name", previous_response_id=response.response_metadata["id"]
    )
    assert isinstance(second_response.content, list)
    assert "bobo" in second_response.content[0]["text"].lower()  # type: ignore


def test_file_search() -> None:
    pytest.skip()  # TODO: set up infra
    llm = ChatOpenAI(model="gpt-4o-mini")
    tool = {
        "type": "file_search",
        "vector_store_ids": [os.environ["OPENAI_VECTOR_STORE_ID"]],
    }
    response = llm.invoke("What is deep research by OpenAI?", tools=[tool])
    _check_response(response)

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("What is deep research by OpenAI?", tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)
