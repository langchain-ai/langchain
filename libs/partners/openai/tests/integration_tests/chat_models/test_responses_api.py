"""Test Responses API usage."""

import json
import os
from typing import Annotated, Any, Optional, cast

import openai
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"


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
    assert response.response_metadata["service_tier"]
    for tool_output in response.additional_kwargs["tool_outputs"]:
        assert tool_output["id"]
        assert tool_output["status"]
        assert tool_output["type"]


@pytest.mark.flaky(retries=3, delay=1)
def test_web_search() -> None:
    llm = ChatOpenAI(model=MODEL_NAME)
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
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What was a positive news story from today?",
                    }
                ],
            },
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


@pytest.mark.flaky(retries=3, delay=1)
async def test_web_search_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME)
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


@pytest.mark.flaky(retries=3, delay=1)
def test_function_calling() -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME)
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

    response = bound_llm.invoke("What was a positive news story from today?")
    _check_response(response)


class Foo(BaseModel):
    response: str


class FooDict(TypedDict):
    response: str


def test_parsed_pydantic_schema() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text()))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text()))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response


async def test_parsed_pydantic_schema_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text()))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text()))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("schema", [Foo.model_json_schema(), FooDict])
def test_parsed_dict_schema(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are ya", response_format=schema)
    parsed = json.loads(response.text())
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"] and isinstance(parsed["response"], str)

    # Test stream
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text())
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"] and isinstance(parsed["response"], str)


def test_parsed_strict() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)

    class InvalidJoke(TypedDict):
        setup: Annotated[str, ..., "The setup of the joke"]
        punchline: Annotated[str, None, "The punchline of the joke"]

    # Test not strict
    response = llm.invoke("Tell me a joke", response_format=InvalidJoke)
    parsed = json.loads(response.text())
    assert parsed == response.additional_kwargs["parsed"]

    # Test strict
    with pytest.raises(openai.BadRequestError):
        llm.invoke(
            "Tell me a joke about cats.", response_format=InvalidJoke, strict=True
        )
    with pytest.raises(openai.BadRequestError):
        next(
            llm.stream(
                "Tell me a joke about cats.", response_format=InvalidJoke, strict=True
            )
        )


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("schema", [Foo.model_json_schema(), FooDict])
async def test_parsed_dict_schema_async(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=schema)
    parsed = json.loads(response.text())
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"] and isinstance(parsed["response"], str)

    # Test stream
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text())
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"] and isinstance(parsed["response"], str)


def test_function_calling_and_structured_output() -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    bound_llm = llm.bind_tools([multiply], response_format=Foo, strict=True)
    # Test structured output
    response = llm.invoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text()))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test function calling
    ai_msg = cast(AIMessage, bound_llm.invoke("whats 5 * 4"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}


def test_reasoning() -> None:
    llm = ChatOpenAI(model="o3-mini", use_responses_api=True)
    response = llm.invoke("Hello", reasoning={"effort": "low"})
    assert isinstance(response, AIMessage)
    assert response.additional_kwargs["reasoning"]

    # Test init params + streaming
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="low", use_responses_api=True)
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("Hello"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)
    assert full.additional_kwargs["reasoning"]


def test_stateful_api() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are you, my name is Bobo")
    assert "id" in response.response_metadata

    second_response = llm.invoke(
        "what's my name", previous_response_id=response.response_metadata["id"]
    )
    assert isinstance(second_response.content, list)
    assert "bobo" in second_response.content[0]["text"].lower()  # type: ignore


def test_route_from_model_kwargs() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, model_kwargs={"truncation": "auto"})
    _ = next(llm.stream("Hello"))


@pytest.mark.flaky(retries=3, delay=1)
def test_computer_calls() -> None:
    llm = ChatOpenAI(model="computer-use-preview", model_kwargs={"truncation": "auto"})
    tool = {
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser",
    }
    llm_with_tools = llm.bind_tools([tool], tool_choice="any")
    response = llm_with_tools.invoke("Please open the browser.")
    assert response.additional_kwargs["tool_outputs"]


def test_file_search() -> None:
    pytest.skip()  # TODO: set up infra
    llm = ChatOpenAI(model=MODEL_NAME)
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


def test_stream_reasoning_summary() -> None:
    reasoning = {"effort": "medium", "summary": "auto"}

    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, model_kwargs={"reasoning": reasoning}
    )
    message_1 = {"role": "user", "content": "What is 3^3?"}
    response_1: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([message_1]):
        assert isinstance(chunk, AIMessageChunk)
        response_1 = chunk if response_1 is None else response_1 + chunk
    assert isinstance(response_1, AIMessageChunk)
    reasoning = response_1.additional_kwargs["reasoning"]
    assert set(reasoning.keys()) == {"id", "type", "summary"}
    summary = reasoning["summary"]
    assert isinstance(summary, list)
    for block in summary:
        assert isinstance(block, dict)
        assert isinstance(block["type"], str)
        assert isinstance(block["text"], str)
        assert block["text"]

    # Check we can pass back summaries
    message_2 = {"role": "user", "content": "Thank you."}
    response_2 = llm.invoke([message_1, response_1, message_2])
    assert isinstance(response_2, AIMessage)
