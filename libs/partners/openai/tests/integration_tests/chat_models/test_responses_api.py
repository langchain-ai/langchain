"""Test Responses API usage."""

import json
import os
from typing import Annotated, Any, Literal, Optional, Union, cast

import openai
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    MessageLikeRepresentation,
)
from langchain_core.messages.v1 import AIMessage as AIMessageV1
from langchain_core.messages.v1 import AIMessageChunk as AIMessageChunkV1
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, ChatOpenAIV1

MODEL_NAME = "gpt-4o-mini"


def _check_response(
    response: Optional[Union[BaseMessage, AIMessageV1]], output_version: str
) -> None:
    if output_version == "v1":
        assert isinstance(response, AIMessageV1) or isinstance(
            response, AIMessageChunkV1
        )
    else:
        assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "text":
            assert isinstance(block["text"], str)  # type: ignore[typeddict-item]
            for annotation in block["annotations"]:  # type: ignore[typeddict-item]
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
                elif annotation["type"] == "citation":
                    assert all(key in annotation for key in ["title", "type"])
                    if "url" in annotation:
                        assert "start_index" in annotation
                        assert "end_index" in annotation

    if output_version == "v1":
        text_content = response.text
    else:
        text_content = response.text()  # type: ignore[operator,misc]
    assert isinstance(text_content, str)
    assert text_content
    assert response.usage_metadata
    assert response.usage_metadata["input_tokens"] > 0
    assert response.usage_metadata["output_tokens"] > 0
    assert response.usage_metadata["total_tokens"] > 0
    assert response.response_metadata["model_name"]
    assert response.response_metadata["service_tier"]  # type: ignore[typeddict-item]


@pytest.mark.default_cassette("test_web_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_web_search(output_version: Literal["responses/v1", "v1"]) -> None:
    if output_version == "v1":
        llm = ChatOpenAIV1(model=MODEL_NAME)
    else:
        llm = ChatOpenAI(model=MODEL_NAME, output_version=output_version)  # type: ignore[assignment]
    first_response = llm.invoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(first_response, output_version)

    # Test streaming
    if isinstance(llm, ChatOpenAIV1):
        full: Optional[AIMessageChunkV1] = None
        for chunk in llm.stream(
            "What was a positive news story from today?",
            tools=[{"type": "web_search_preview"}],
        ):
            assert isinstance(chunk, AIMessageChunkV1)
            full = chunk if full is None else full + chunk
    else:
        full: Optional[BaseMessageChunk] = None  # type: ignore[no-redef]
        for chunk in llm.stream(
            "What was a positive news story from today?",
            tools=[{"type": "web_search_preview"}],
        ):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
    _check_response(full, output_version)

    # Use OpenAI's stateful API
    response = llm.invoke(
        "what about a negative one",
        tools=[{"type": "web_search_preview"}],
        previous_response_id=first_response.response_metadata["id"],  # type: ignore[typeddict-item]
    )
    _check_response(response, output_version)

    # Manually pass in chat history
    response = llm.invoke(
        [
            {"role": "user", "content": "What was a positive news story from today?"},
            first_response,
            {"role": "user", "content": "what about a negative one"},
        ],
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response, output_version)

    # Bind tool
    response = llm.bind_tools([{"type": "web_search_preview"}]).invoke(
        "What was a positive news story from today?"
    )
    _check_response(response, output_version)

    for msg in [first_response, full, response]:
        assert msg is not None
        block_types = [block["type"] for block in msg.content]  # type: ignore[index]
        if output_version == "responses/v1":
            assert block_types == ["web_search_call", "text"]
        else:
            assert block_types == ["web_search_call", "web_search_result", "text"]


@pytest.mark.flaky(retries=3, delay=1)
async def test_web_search_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME)
    response = await llm.ainvoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response, "v0")
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
    _check_response(full, "v0")

    for msg in [response, full]:
        assert msg.additional_kwargs["tool_outputs"]
        assert len(msg.additional_kwargs["tool_outputs"]) == 1
        tool_output = msg.additional_kwargs["tool_outputs"][0]
        assert tool_output["type"] == "web_search_call"


@pytest.mark.default_cassette("test_function_calling.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_function_calling(output_version: Literal["v0", "responses/v1"]) -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME, output_version=output_version)
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
    _check_response(response, output_version)


@pytest.mark.default_cassette("test_function_calling.yaml.gz")
@pytest.mark.vcr
def test_function_calling_v1() -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAIV1(model=MODEL_NAME)
    bound_llm = llm.bind_tools([multiply, {"type": "web_search_preview"}])
    ai_msg = bound_llm.invoke("whats 5 * 4")
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}

    full: Any = None
    for chunk in bound_llm.stream("whats 5 * 4"):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk
    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["name"] == "multiply"
    assert set(full.tool_calls[0]["args"]) == {"x", "y"}

    response = bound_llm.invoke("What was a positive news story from today?")
    _check_response(response, "v1")


class Foo(BaseModel):
    response: str


class FooDict(TypedDict):
    response: str


@pytest.mark.default_cassette("test_parsed_pydantic_schema.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_parsed_pydantic_schema(output_version: Literal["v0", "responses/v1"]) -> None:
    llm = ChatOpenAI(
        model=MODEL_NAME, use_responses_api=True, output_version=output_version
    )
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


@pytest.mark.default_cassette("test_parsed_pydantic_schema.yaml.gz")
@pytest.mark.vcr
def test_parsed_pydantic_schema_v1() -> None:
    llm = ChatOpenAIV1(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are ya", response_format=Foo)
    assert response.text
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.parsed
    assert parsed.response

    # Test stream
    full: Optional[AIMessageChunkV1] = None
    chunks = []
    for chunk in llm.stream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk
        chunks.append(chunk)
    assert isinstance(full, AIMessageChunkV1)
    assert full.text
    parsed = Foo(**json.loads(full.text))
    assert parsed == full.parsed
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


@pytest.mark.default_cassette("test_reasoning.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_reasoning(output_version: Literal["v0", "responses/v1"]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    response = llm.invoke("Hello", reasoning={"effort": "low"})
    assert isinstance(response, AIMessage)

    # Test init params + streaming
    llm = ChatOpenAI(
        model="o4-mini", reasoning={"effort": "low"}, output_version=output_version
    )
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("Hello"):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)

    for msg in [response, full]:
        if output_version == "v0":
            assert msg.additional_kwargs["reasoning"]
        else:
            block_types = [block["type"] for block in msg.content]
            assert block_types == ["reasoning", "text"]


@pytest.mark.default_cassette("test_reasoning.yaml.gz")
@pytest.mark.vcr
def test_reasoning_v1() -> None:
    llm = ChatOpenAIV1(model="o4-mini", use_responses_api=True)
    response = llm.invoke("Hello", reasoning={"effort": "low"})
    assert isinstance(response, AIMessageV1)

    # Test init params + streaming
    llm = ChatOpenAIV1(model="o4-mini", reasoning={"effort": "low"})
    full: Optional[AIMessageChunkV1] = None
    for chunk in llm.stream("Hello"):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunkV1)

    for msg in [response, full]:
        block_types = [block["type"] for block in msg.content]
        assert block_types == ["reasoning", "text"]


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
    llm = ChatOpenAI(
        model=MODEL_NAME, model_kwargs={"text": {"format": {"type": "text"}}}
    )
    _ = next(llm.stream("Hello"))


@pytest.mark.flaky(retries=3, delay=1)
def test_computer_calls() -> None:
    llm = ChatOpenAI(model="computer-use-preview", truncation="auto")
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
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    tool = {
        "type": "file_search",
        "vector_store_ids": [os.environ["OPENAI_VECTOR_STORE_ID"]],
    }

    input_message = {"role": "user", "content": "What is deep research by OpenAI?"}
    response = llm.invoke([input_message], tools=[tool])
    _check_response(response, "v0")

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message], tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full, "v0")

    next_message = {"role": "user", "content": "Thank you."}
    _ = llm.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_stream_reasoning_summary.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_stream_reasoning_summary(
    output_version: Literal["v0", "responses/v1"],
) -> None:
    llm = ChatOpenAI(
        model="o4-mini",
        # Routes to Responses API if `reasoning` is set.
        reasoning={"effort": "medium", "summary": "auto"},
        output_version=output_version,
    )
    message_1 = {
        "role": "user",
        "content": "What was the third tallest buliding in the year 2000?",
    }
    response_1: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([message_1]):
        assert isinstance(chunk, AIMessageChunk)
        response_1 = chunk if response_1 is None else response_1 + chunk
    assert isinstance(response_1, AIMessageChunk)
    if output_version == "v0":
        reasoning = response_1.additional_kwargs["reasoning"]
        assert set(reasoning.keys()) == {"id", "type", "summary"}
        summary = reasoning["summary"]
        assert isinstance(summary, list)
        for block in summary:
            assert isinstance(block, dict)
            assert isinstance(block["type"], str)
            assert isinstance(block["text"], str)
            assert block["text"]
    else:
        # output_version == "responses/v1"
        reasoning = next(
            block
            for block in response_1.content
            if block["type"] == "reasoning"  # type: ignore[index]
        )
        assert set(reasoning.keys()) == {"id", "type", "summary", "index"}
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


@pytest.mark.default_cassette("test_stream_reasoning_summary.yaml.gz")
@pytest.mark.vcr
def test_stream_reasoning_summary_v1() -> None:
    llm = ChatOpenAIV1(
        model="o4-mini",
        # Routes to Responses API if `reasoning` is set.
        reasoning={"effort": "medium", "summary": "auto"},
    )
    message_1 = {
        "role": "user",
        "content": "What was the third tallest buliding in the year 2000?",
    }
    response_1: Optional[AIMessageChunkV1] = None
    for chunk in llm.stream([message_1]):
        assert isinstance(chunk, AIMessageChunkV1)
        response_1 = chunk if response_1 is None else response_1 + chunk
    assert isinstance(response_1, AIMessageChunkV1)

    total_reasoning_blocks = 0
    for block in response_1.content:
        if block["type"] == "reasoning":
            total_reasoning_blocks += 1
            assert isinstance(block["id"], str) and block["id"].startswith("rs_")
            assert isinstance(block["reasoning"], str)
            assert isinstance(block["index"], int)
    assert (
        total_reasoning_blocks > 1
    )  # This query typically generates multiple reasoning blocks

    # Check we can pass back summaries
    message_2 = {"role": "user", "content": "Thank you."}
    response_2 = llm.invoke([message_1, response_1, message_2])
    assert isinstance(response_2, AIMessageV1)


@pytest.mark.default_cassette("test_code_interpreter.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_code_interpreter(output_version: Literal["v0", "responses/v1"]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": {"type": "auto"}}]
    )
    input_message = {
        "role": "user",
        "content": "Write and run code to answer the question: what is 3^3?",
    }
    response = llm_with_tools.invoke([input_message])
    assert isinstance(response, AIMessage)
    _check_response(response, output_version)
    if output_version == "v0":
        tool_outputs = [
            item
            for item in response.additional_kwargs["tool_outputs"]
            if item["type"] == "code_interpreter_call"
        ]
    else:
        # responses/v1
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
    assert len(tool_outputs) == 1

    # Test streaming
    # Use same container
    container_id = tool_outputs[0]["container_id"]
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": container_id}]
    )

    full: Optional[BaseMessageChunk] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    if output_version == "v0":
        tool_outputs = [
            item
            for item in response.additional_kwargs["tool_outputs"]
            if item["type"] == "code_interpreter_call"
        ]
    else:
        # responses/v1
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
    assert tool_outputs

    # Test we can pass back in
    next_message = {"role": "user", "content": "Please add more comments to the code."}
    _ = llm_with_tools.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_code_interpreter.yaml.gz")
@pytest.mark.vcr
def test_code_interpreter_v1() -> None:
    llm = ChatOpenAIV1(model="o4-mini", use_responses_api=True)
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": {"type": "auto"}}]
    )
    input_message = {
        "role": "user",
        "content": "Write and run code to answer the question: what is 3^3?",
    }
    response = llm_with_tools.invoke([input_message])
    assert isinstance(response, AIMessageV1)
    _check_response(response, "v1")

    tool_outputs = [
        item for item in response.content if item["type"] == "code_interpreter_call"
    ]
    code_interpreter_result = next(
        item for item in response.content if item["type"] == "code_interpreter_result"
    )
    assert tool_outputs
    assert code_interpreter_result
    assert len(tool_outputs) == 1

    # Test streaming
    # Use same container
    container_id = tool_outputs[0]["container_id"]  # type: ignore[typeddict-item]
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": container_id}]
    )

    full: Optional[AIMessageChunkV1] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunkV1)
    code_interpreter_call = next(
        item for item in full.content if item["type"] == "code_interpreter_call"
    )
    code_interpreter_result = next(
        item for item in full.content if item["type"] == "code_interpreter_result"
    )
    assert code_interpreter_call
    assert code_interpreter_result
    assert tool_outputs

    # Test we can pass back in
    next_message = {"role": "user", "content": "Please add more comments to the code."}
    _ = llm_with_tools.invoke([input_message, full, next_message])


@pytest.mark.vcr
def test_mcp_builtin() -> None:
    llm = ChatOpenAI(model="o4-mini", use_responses_api=True)

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": {"always": {"tool_names": ["read_wiki_structure"]}},
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP spec "
            "support?"
        ),
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)

    approval_message = HumanMessage(
        [
            {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": output["id"],
            }
            for output in response.additional_kwargs["tool_outputs"]
            if output["type"] == "mcp_approval_request"
        ]
    )
    _ = llm_with_tools.invoke(
        [approval_message], previous_response_id=response.response_metadata["id"]
    )


@pytest.mark.vcr
def test_mcp_builtin_zdr() -> None:
    llm = ChatOpenAI(
        model="o4-mini",
        output_version="responses/v1",
        store=False,
        include=["reasoning.encrypted_content"],
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": {"always": {"tool_names": ["read_wiki_structure"]}},
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP spec "
            "support?"
        ),
    }
    full: Optional[BaseMessageChunk] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert all(isinstance(block, dict) for block in full.content)

    approval_message = HumanMessage(
        [
            {
                "type": "mcp_approval_response",
                "approve": True,
                "approval_request_id": block["id"],  # type: ignore[index]
            }
            for block in full.content
            if block["type"] == "mcp_approval_request"  # type: ignore[index]
        ]
    )
    _ = llm_with_tools.invoke([input_message, full, approval_message])


@pytest.mark.default_cassette("test_mcp_builtin_zdr.yaml.gz")
@pytest.mark.vcr
def test_mcp_builtin_zdr_v1() -> None:
    llm = ChatOpenAIV1(
        model="o4-mini", store=False, include=["reasoning.encrypted_content"]
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": {"always": {"tool_names": ["read_wiki_structure"]}},
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP spec "
            "support?"
        ),
    }
    full: Optional[AIMessageChunkV1] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunkV1)
    assert all(isinstance(block, dict) for block in full.content)

    approval_message = HumanMessageV1(
        [
            {
                "type": "non_standard",
                "value": {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": block["value"]["id"],  # type: ignore[index]
                },
            }
            for block in full.content
            if block["type"] == "non_standard"
            and block["value"]["type"] == "mcp_approval_request"  # type: ignore[index]
        ]
    )
    _ = llm_with_tools.invoke([input_message, full, approval_message])


@pytest.mark.default_cassette("test_image_generation_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_image_generation_streaming(
    output_version: Literal["v0", "responses/v1"],
) -> None:
    """Test image generation streaming."""
    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version=output_version
    )
    tool = {
        "type": "image_generation",
        # For testing purposes let's keep the quality low, so the test runs faster.
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }

    # Example tool output for an image
    # {
    #     "background": "opaque",
    #     "id": "ig_683716a8ddf0819888572b20621c7ae4029ec8c11f8dacf8",
    #     "output_format": "png",
    #     "quality": "high",
    #     "revised_prompt": "A fluffy, fuzzy cat sitting calmly, with soft fur, bright "
    #     "eyes, and a cute, friendly expression. The background is "
    #     "simple and light to emphasize the cat's texture and "
    #     "fluffiness.",
    #     "size": "1024x1024",
    #     "status": "completed",
    #     "type": "image_generation_call",
    #     "result": # base64 encode image data
    # }

    expected_keys = {
        "id",
        "index",
        "background",
        "output_format",
        "quality",
        "result",
        "revised_prompt",
        "size",
        "status",
        "type",
    }

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("Draw a random short word in green font.", tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    complete_ai_message = cast(AIMessageChunk, full)
    # At the moment, the streaming API does not pick up annotations fully.
    # So the following check is commented out.
    # _check_response(complete_ai_message)
    if output_version == "v0":
        assert complete_ai_message.additional_kwargs["tool_outputs"]
        tool_output = complete_ai_message.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    elif output_version == "responses/v1":
        tool_output = next(
            block
            for block in complete_ai_message.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        # v1
        standard_keys = {"type", "base64", "id", "status", "index"}
        tool_output = next(
            block
            for block in complete_ai_message.content
            if isinstance(block, dict) and block["type"] == "image"
        )
        assert set(standard_keys).issubset(tool_output.keys())


@pytest.mark.default_cassette("test_image_generation_streaming.yaml.gz")
@pytest.mark.vcr
def test_image_generation_streaming_v1() -> None:
    """Test image generation streaming."""
    llm = ChatOpenAIV1(model="gpt-4.1", use_responses_api=True)
    tool = {
        "type": "image_generation",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }

    expected_keys = {
        # Standard
        "type",
        "base64",
        "mime_type",
        "id",
        "index",
        # OpenAI-specific
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    full: Optional[AIMessageChunkV1] = None
    for chunk in llm.stream("Draw a random short word in green font.", tools=[tool]):
        assert isinstance(chunk, AIMessageChunkV1)
        full = chunk if full is None else full + chunk
    complete_ai_message = cast(AIMessageChunkV1, full)

    tool_output = next(
        block
        for block in complete_ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(expected_keys).issubset(tool_output.keys())


@pytest.mark.default_cassette("test_image_generation_multi_turn.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_image_generation_multi_turn(
    output_version: Literal["v0", "responses/v1"],
) -> None:
    """Test multi-turn editing of image generation by passing in history."""
    # Test multi-turn
    llm = ChatOpenAI(
        model="gpt-4.1", use_responses_api=True, output_version=output_version
    )
    # Test invocation
    tool = {
        "type": "image_generation",
        # For testing purposes let's keep the quality low, so the test runs faster.
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }
    llm_with_tools = llm.bind_tools([tool])

    chat_history: list[MessageLikeRepresentation] = [
        {"role": "user", "content": "Draw a random short word in green font."}
    ]
    ai_message = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message, AIMessage)
    _check_response(ai_message, output_version)

    expected_keys = {
        "id",
        "background",
        "output_format",
        "quality",
        "result",
        "revised_prompt",
        "size",
        "status",
        "type",
    }

    if output_version == "v0":
        tool_output = ai_message.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    elif output_version == "responses/v1":
        tool_output = next(
            block
            for block in ai_message.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        standard_keys = {"type", "base64", "id", "status"}
        tool_output = next(
            block
            for block in ai_message.content
            if isinstance(block, dict) and block["type"] == "image"
        )
        assert set(standard_keys).issubset(tool_output.keys())

    # Example tool output for an image (v0)
    # {
    #     "background": "opaque",
    #     "id": "ig_683716a8ddf0819888572b20621c7ae4029ec8c11f8dacf8",
    #     "output_format": "png",
    #     "quality": "high",
    #     "revised_prompt": "A fluffy, fuzzy cat sitting calmly, with soft fur, bright "
    #     "eyes, and a cute, friendly expression. The background is "
    #     "simple and light to emphasize the cat's texture and "
    #     "fluffiness.",
    #     "size": "1024x1024",
    #     "status": "completed",
    #     "type": "image_generation_call",
    #     "result": # base64 encode image data
    # }

    chat_history.extend(
        [
            # AI message with tool output
            ai_message,
            # New request
            {
                "role": "user",
                "content": (
                    "Now, change the font to blue. Keep the word and everything else "
                    "the same."
                ),
            },
        ]
    )

    ai_message2 = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message2, AIMessage)
    _check_response(ai_message2, output_version)

    if output_version == "v0":
        tool_output = ai_message2.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    elif output_version == "responses/v1":
        tool_output = next(
            block
            for block in ai_message2.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        standard_keys = {"type", "base64", "id", "status"}
        tool_output = next(
            block
            for block in ai_message2.content
            if isinstance(block, dict) and block["type"] == "image"
        )
        assert set(standard_keys).issubset(tool_output.keys())


@pytest.mark.default_cassette("test_image_generation_multi_turn.yaml.gz")
@pytest.mark.vcr
def test_image_generation_multi_turn_v1() -> None:
    """Test multi-turn editing of image generation by passing in history."""
    # Test multi-turn
    llm = ChatOpenAIV1(model="gpt-4.1", use_responses_api=True)
    # Test invocation
    tool = {
        "type": "image_generation",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }
    llm_with_tools = llm.bind_tools([tool])

    chat_history: list[MessageLikeRepresentation] = [
        {"role": "user", "content": "Draw a random short word in green font."}
    ]
    ai_message = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message, AIMessageV1)
    _check_response(ai_message, "v1")

    expected_keys = {
        # Standard
        "type",
        "base64",
        "mime_type",
        "id",
        # OpenAI-specific
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    standard_keys = {"type", "base64", "id", "status"}
    tool_output = next(
        block
        for block in ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())

    chat_history.extend(
        [
            # AI message with tool output
            ai_message,
            # New request
            {
                "role": "user",
                "content": (
                    "Now, change the font to blue. Keep the word and everything else "
                    "the same."
                ),
            },
        ]
    )

    ai_message2 = llm_with_tools.invoke(chat_history)
    assert isinstance(ai_message2, AIMessageV1)
    _check_response(ai_message2, "v1")

    tool_output = next(
        block
        for block in ai_message2.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(expected_keys).issubset(tool_output.keys())
