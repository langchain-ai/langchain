"""Test Responses API usage."""

import base64
import json
import os
from typing import Annotated, Any, Literal, cast

import openai
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, custom_tool
from langchain_openai.chat_models.base import _convert_to_openai_response_format

MODEL_NAME = "gpt-4o-mini"


def _check_response(response: BaseMessage | None) -> None:
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "text":
            assert isinstance(block.get("text"), str)
            annotations = block.get("annotations", [])
            for annotation in annotations:
                if annotation["type"] == "file_citation":
                    assert all(
                        key in annotation
                        for key in ["file_id", "filename", "file_index", "type"]
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
    text_content = response.text  # type: ignore[operator,misc]
    assert isinstance(text_content, str)
    assert text_content
    assert response.usage_metadata
    assert response.usage_metadata["input_tokens"] > 0
    assert response.usage_metadata["output_tokens"] > 0
    assert response.usage_metadata["total_tokens"] > 0
    assert response.response_metadata["model_name"]
    assert response.response_metadata["service_tier"]  # type: ignore[typeddict-item]


@pytest.mark.vcr
def test_incomplete_response() -> None:
    model = ChatOpenAI(
        model=MODEL_NAME, use_responses_api=True, max_completion_tokens=16
    )
    response = model.invoke("Tell me a 100 word story about a bear.")
    assert response.response_metadata["incomplete_details"]
    assert response.response_metadata["incomplete_details"]["reason"]
    assert response.response_metadata["status"] == "incomplete"

    full: AIMessageChunk | None = None
    for chunk in model.stream("Tell me a 100 word story about a bear."):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata["incomplete_details"]
    assert full.response_metadata["incomplete_details"]["reason"]
    assert full.response_metadata["status"] == "incomplete"


@pytest.mark.default_cassette("test_web_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_web_search(output_version: Literal["responses/v1", "v1"]) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, output_version=output_version)
    first_response = llm.invoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(first_response)

    # Test streaming
    full: BaseMessageChunk | None = None
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
            {"role": "user", "content": "What was a positive news story from today?"},
            first_response,
            {"role": "user", "content": "what about a negative one"},
        ],
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)

    # Bind tool
    response = llm.bind_tools([{"type": "web_search_preview"}]).invoke(
        "What was a positive news story from today?"
    )
    _check_response(response)

    for msg in [first_response, full, response]:
        assert msg is not None
        block_types = [block["type"] for block in msg.content]  # type: ignore[index]
        if output_version == "responses/v1":
            assert block_types == ["web_search_call", "text"]
        else:
            assert block_types == ["server_tool_call", "server_tool_result", "text"]


@pytest.mark.flaky(retries=3, delay=1)
async def test_web_search_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, output_version="v0")
    response = await llm.ainvoke(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    )
    _check_response(response)
    assert response.response_metadata["status"]

    # Test streaming
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream(
        "What was a positive news story from today?",
        tools=[{"type": "web_search_preview"}],
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)

    for msg in [response, full]:
        assert msg.additional_kwargs["tool_outputs"]
        assert len(msg.additional_kwargs["tool_outputs"]) == 1
        tool_output = msg.additional_kwargs["tool_outputs"][0]
        assert tool_output["type"] == "web_search_call"


@pytest.mark.default_cassette("test_function_calling.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_function_calling(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
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

    for msg in [ai_msg, full]:
        assert len(msg.content_blocks) == 1
        assert msg.content_blocks[0]["type"] == "tool_call"

    response = bound_llm.invoke("What was a positive news story from today?")
    _check_response(response)


@pytest.mark.default_cassette("test_agent_loop.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_agent_loop(output_version: Literal["responses/v1", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.default_cassette("test_agent_loop_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_agent_loop_streaming(output_version: Literal["responses/v1", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatOpenAI(
        model="gpt-5.2",
        use_responses_api=True,
        reasoning={"effort": "medium", "summary": "auto"},
        streaming=True,
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


class Foo(BaseModel):
    response: str


class FooDict(TypedDict):
    response: str


@pytest.mark.default_cassette("test_parsed_pydantic_schema.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_parsed_pydantic_schema(
    output_version: Literal["v0", "responses/v1", "v1"],
) -> None:
    llm = ChatOpenAI(
        model=MODEL_NAME, use_responses_api=True, output_version=output_version
    )
    response = llm.invoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response


async def test_parsed_pydantic_schema_async() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=Foo)
    parsed = Foo(**json.loads(response.text))
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed.response

    # Test stream
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("how are ya", response_format=Foo):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = Foo(**json.loads(full.text))
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed.response


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("schema", [Foo.model_json_schema(), FooDict])
def test_parsed_dict_schema(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = llm.invoke("how are ya", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

    # Test stream
    full: BaseMessageChunk | None = None
    for chunk in llm.stream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text)
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)


def test_parsed_strict() -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)

    class Joke(TypedDict):
        setup: Annotated[str, ..., "The setup of the joke"]
        punchline: Annotated[str, None, "The punchline of the joke"]

    schema = _convert_to_openai_response_format(Joke)
    invalid_schema = cast(dict, _convert_to_openai_response_format(Joke, strict=True))
    invalid_schema["json_schema"]["schema"]["required"] = ["setup"]  # make invalid

    # Test not strict
    response = llm.invoke("Tell me a joke", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]

    # Test strict
    with pytest.raises(openai.BadRequestError):
        llm.invoke(
            "Tell me a joke about cats.", response_format=invalid_schema, strict=True
        )
    with pytest.raises(openai.BadRequestError):
        next(
            llm.stream(
                "Tell me a joke about cats.",
                response_format=invalid_schema,
                strict=True,
            )
        )


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("schema", [Foo.model_json_schema(), FooDict])
async def test_parsed_dict_schema_async(schema: Any) -> None:
    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    response = await llm.ainvoke("how are ya", response_format=schema)
    parsed = json.loads(response.text)
    assert parsed == response.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)

    # Test stream
    full: BaseMessageChunk | None = None
    async for chunk in llm.astream("how are ya", response_format=schema):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    parsed = json.loads(full.text)
    assert parsed == full.additional_kwargs["parsed"]
    assert parsed["response"]
    assert isinstance(parsed["response"], str)


@pytest.mark.parametrize("schema", [Foo, Foo.model_json_schema(), FooDict])
def test_function_calling_and_structured_output(schema: Any) -> None:
    def multiply(x: int, y: int) -> int:
        """return x * y"""
        return x * y

    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)
    bound_llm = llm.bind_tools([multiply], response_format=schema, strict=True)
    # Test structured output
    response = llm.invoke("how are ya", response_format=schema)
    if schema == Foo:
        parsed = schema(**json.loads(response.text))
        assert parsed.response
    else:
        parsed = json.loads(response.text)
        assert parsed["response"]
    assert parsed == response.additional_kwargs["parsed"]

    # Test function calling
    ai_msg = cast(AIMessage, bound_llm.invoke("whats 5 * 4"))
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "multiply"
    assert set(ai_msg.tool_calls[0]["args"]) == {"x", "y"}


@pytest.mark.default_cassette("test_reasoning.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_reasoning(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    response = llm.invoke("Hello", reasoning={"effort": "low"})
    assert isinstance(response, AIMessage)

    # Test init params + streaming
    llm = ChatOpenAI(
        model="o4-mini", reasoning={"effort": "low"}, output_version=output_version
    )
    full: BaseMessageChunk | None = None
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
    llm = ChatOpenAI(
        model="computer-use-preview", truncation="auto", output_version="v0"
    )
    tool = {
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser",
    }
    llm_with_tools = llm.bind_tools([tool], tool_choice="any")
    response = llm_with_tools.invoke("Please open the browser.")
    assert response.additional_kwargs["tool_outputs"]


@pytest.mark.default_cassette("test_file_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_file_search(
    output_version: Literal["responses/v1", "v1"],
) -> None:
    vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        pytest.skip()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        use_responses_api=True,
        output_version=output_version,
    )
    tool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_id],
    }

    input_message = {"role": "user", "content": "What is deep research by OpenAI?"}
    response = llm.invoke([input_message], tools=[tool])
    _check_response(response)

    if output_version == "v1":
        assert [block["type"] for block in response.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "text",
        ]
    else:
        assert [block["type"] for block in response.content] == [  # type: ignore[index]
            "file_search_call",
            "text",
        ]

    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message], tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_response(full)

    if output_version == "v1":
        assert [block["type"] for block in full.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "text",
        ]
    else:
        assert [block["type"] for block in full.content] == ["file_search_call", "text"]  # type: ignore[index]

    next_message = {"role": "user", "content": "Thank you."}
    _ = llm.invoke([input_message, full, next_message])

    for message in [response, full]:
        assert [block["type"] for block in message.content_blocks] == [
            "server_tool_call",
            "server_tool_result",
            "text",
        ]


@pytest.mark.default_cassette("test_stream_reasoning_summary.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_stream_reasoning_summary(
    output_version: Literal["v0", "responses/v1", "v1"],
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
    response_1: BaseMessageChunk | None = None
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
    elif output_version == "responses/v1":
        reasoning = next(
            block
            for block in response_1.content
            if block["type"] == "reasoning"  # type: ignore[index]
        )
        if isinstance(reasoning, str):
            reasoning = json.loads(reasoning)
        assert set(reasoning.keys()) == {"id", "type", "summary", "index"}
        summary = reasoning["summary"]
        assert isinstance(summary, list)
        for block in summary:
            assert isinstance(block, dict)
            assert isinstance(block["type"], str)
            assert isinstance(block["text"], str)
            assert block["text"]
    else:
        # v1
        total_reasoning_blocks = 0
        for block in response_1.content_blocks:
            if block["type"] == "reasoning":
                total_reasoning_blocks += 1
                assert isinstance(block.get("id"), str)
                assert block.get("id", "").startswith("rs_")
                assert isinstance(block.get("reasoning"), str)
                assert isinstance(block.get("index"), str)
        assert (
            total_reasoning_blocks > 1
        )  # This query typically generates multiple reasoning blocks

    # Check we can pass back summaries
    message_2 = {"role": "user", "content": "Thank you."}
    response_2 = llm.invoke([message_1, response_1, message_2])
    assert isinstance(response_2, AIMessage)


@pytest.mark.default_cassette("test_code_interpreter.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_code_interpreter(output_version: Literal["v0", "responses/v1", "v1"]) -> None:
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
    _check_response(response)
    if output_version == "v0":
        tool_outputs = [
            item
            for item in response.additional_kwargs["tool_outputs"]
            if item["type"] == "code_interpreter_call"
        ]
        assert len(tool_outputs) == 1
    elif output_version == "responses/v1":
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
        assert len(tool_outputs) == 1
    else:
        # v1
        tool_outputs = [
            item
            for item in response.content_blocks
            if item["type"] == "server_tool_call" and item["name"] == "code_interpreter"
        ]
        code_interpreter_result = next(
            item
            for item in response.content_blocks
            if item["type"] == "server_tool_result"
        )
        assert tool_outputs
        assert code_interpreter_result
    assert len(tool_outputs) == 1

    # Test streaming
    # Use same container
    container_id = tool_outputs[0].get("container_id") or tool_outputs[0].get(
        "extras", {}
    ).get("container_id")
    llm_with_tools = llm.bind_tools(
        [{"type": "code_interpreter", "container": container_id}]
    )

    full: BaseMessageChunk | None = None
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
        assert tool_outputs
    elif output_version == "responses/v1":
        tool_outputs = [
            item
            for item in response.content
            if isinstance(item, dict) and item["type"] == "code_interpreter_call"
        ]
        assert tool_outputs
    else:
        # v1
        code_interpreter_call = next(
            item
            for item in full.content_blocks
            if item["type"] == "server_tool_call" and item["name"] == "code_interpreter"
        )
        code_interpreter_result = next(
            item for item in full.content_blocks if item["type"] == "server_tool_result"
        )
        assert code_interpreter_call
        assert code_interpreter_result

    # Test we can pass back in
    next_message = {"role": "user", "content": "Please add more comments to the code."}
    _ = llm_with_tools.invoke([input_message, full, next_message])


@pytest.mark.vcr
def test_mcp_builtin() -> None:
    llm = ChatOpenAI(model="o4-mini", use_responses_api=True, output_version="v0")

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
        model="gpt-5-nano",
        use_responses_api=True,
        store=False,
        include=["reasoning.encrypted_content"],
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "allowed_tools": ["ask_question"],
                "require_approval": "always",
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP "
            "spec (modelcontextprotocol/modelcontextprotocol) support?"
        ),
    }
    full: BaseMessageChunk | None = None
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
    result = llm_with_tools.invoke([input_message, full, approval_message])
    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm_with_tools.invoke(
        [input_message, full, approval_message, result, next_message]
    )


@pytest.mark.default_cassette("test_mcp_builtin_zdr.yaml.gz")
@pytest.mark.vcr
def test_mcp_builtin_zdr_v1() -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        output_version="v1",
        store=False,
        include=["reasoning.encrypted_content"],
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "allowed_tools": ["ask_question"],
                "require_approval": "always",
            }
        ]
    )
    input_message = {
        "role": "user",
        "content": (
            "What transport protocols does the 2025-03-26 version of the MCP "
            "spec (modelcontextprotocol/modelcontextprotocol) support?"
        ),
    }
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert all(isinstance(block, dict) for block in full.content)

    approval_message = HumanMessage(
        [
            {
                "type": "non_standard",
                "value": {
                    "type": "mcp_approval_response",
                    "approve": True,
                    "approval_request_id": block["value"]["id"],  # type: ignore[index]
                },
            }
            for block in full.content_blocks
            if block["type"] == "non_standard"
            and block["value"]["type"] == "mcp_approval_request"  # type: ignore[index]
        ]
    )
    result = llm_with_tools.invoke([input_message, full, approval_message])
    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm_with_tools.invoke(
        [input_message, full, approval_message, result, next_message]
    )


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

    full: BaseMessageChunk | None = None
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
    else:
        # "responses/v1"
        tool_output = next(
            block
            for block in complete_ai_message.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)


@pytest.mark.default_cassette("test_image_generation_streaming.yaml.gz")
@pytest.mark.vcr
def test_image_generation_streaming_v1() -> None:
    """Test image generation streaming."""
    llm = ChatOpenAI(model="gpt-4.1", use_responses_api=True, output_version="v1")
    tool = {
        "type": "image_generation",
        "quality": "low",
        "output_format": "jpeg",
        "output_compression": 100,
        "size": "1024x1024",
    }

    standard_keys = {"type", "base64", "mime_type", "id", "index"}
    extra_keys = {
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    full: BaseMessageChunk | None = None
    for chunk in llm.stream("Draw a random short word in green font.", tools=[tool]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    complete_ai_message = cast(AIMessageChunk, full)

    tool_output = next(
        block
        for block in complete_ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())


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
    _check_response(ai_message)

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
    _check_response(ai_message2)

    if output_version == "v0":
        tool_output = ai_message2.additional_kwargs["tool_outputs"][0]
        assert set(tool_output.keys()).issubset(expected_keys)
    else:
        # "responses/v1"
        tool_output = next(
            block
            for block in ai_message2.content
            if isinstance(block, dict) and block["type"] == "image_generation_call"
        )
        assert set(tool_output.keys()).issubset(expected_keys)


@pytest.mark.default_cassette("test_image_generation_multi_turn.yaml.gz")
@pytest.mark.vcr
def test_image_generation_multi_turn_v1() -> None:
    """Test multi-turn editing of image generation by passing in history."""
    # Test multi-turn
    llm = ChatOpenAI(model="gpt-4.1", use_responses_api=True, output_version="v1")
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
    assert isinstance(ai_message, AIMessage)
    _check_response(ai_message)

    standard_keys = {"type", "base64", "mime_type", "id"}
    extra_keys = {
        "background",
        "output_format",
        "quality",
        "revised_prompt",
        "size",
        "status",
    }

    tool_output = next(
        block
        for block in ai_message.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())

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
    _check_response(ai_message2)

    tool_output = next(
        block
        for block in ai_message2.content
        if isinstance(block, dict) and block["type"] == "image"
    )
    assert set(standard_keys).issubset(tool_output.keys())
    assert set(extra_keys).issubset(tool_output["extras"].keys())


def test_verbosity_parameter() -> None:
    """Test verbosity parameter with Responses API.

    Tests that the verbosity parameter works correctly with the OpenAI Responses API.

    """
    llm = ChatOpenAI(model=MODEL_NAME, verbosity="medium", use_responses_api=True)
    response = llm.invoke([HumanMessage(content="Hello, explain quantum computing.")])

    assert isinstance(response, AIMessage)
    assert response.content


@pytest.mark.default_cassette("test_custom_tool.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_custom_tool(output_version: Literal["responses/v1", "v1"]) -> None:
    @custom_tool
    def execute_code(code: str) -> str:
        """Execute python code."""
        return "27"

    llm = ChatOpenAI(model="gpt-5", output_version=output_version).bind_tools(
        [execute_code]
    )

    input_message = {"role": "user", "content": "Use the tool to evaluate 3^3."}
    tool_call_message = llm.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    assert len(tool_call_message.tool_calls) == 1
    tool_call = tool_call_message.tool_calls[0]
    tool_message = execute_code.invoke(tool_call)
    response = llm.invoke([input_message, tool_call_message, tool_message])
    assert isinstance(response, AIMessage)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1


@pytest.mark.default_cassette("test_compaction.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_compaction(output_version: Literal["responses/v1", "v1"]) -> None:
    """Test the compaction beta feature."""
    llm = ChatOpenAI(
        model="gpt-5.2",
        context_management=[{"type": "compaction", "compact_threshold": 10_000}],
        output_version=output_version,
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 50000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 50000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    assert third_response.text


@pytest.mark.default_cassette("test_compaction_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_compaction_streaming(output_version: Literal["responses/v1", "v1"]) -> None:
    """Test the compaction beta feature."""
    llm = ChatOpenAI(
        model="gpt-5.2",
        context_management=[{"type": "compaction", "compact_threshold": 10_000}],
        output_version=output_version,
        streaming=True,
    )

    input_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'a' * 50000}",
    }
    messages: list = [input_message]

    first_response = llm.invoke(messages)
    messages.append(first_response)

    second_message = {
        "role": "user",
        "content": f"Generate a one-sentence summary of this:\n\n{'b' * 50000}",
    }
    messages.append(second_message)

    second_response = llm.invoke(messages)
    messages.append(second_response)

    content_blocks = second_response.content_blocks
    compaction_block = next(
        (block for block in content_blocks if block["type"] == "non_standard"),
        None,
    )
    assert compaction_block
    assert compaction_block["value"].get("type") == "compaction"

    third_message = {
        "role": "user",
        "content": "What are we talking about?",
    }
    messages.append(third_message)
    third_response = llm.invoke(messages)
    assert third_response.text


def test_csv_input() -> None:
    """Test CSV file input with both LangChain standard and OpenAI native formats."""
    # Create sample CSV content
    csv_content = (
        "name,age,city\nAlice,30,New York\nBob,25,Los Angeles\nCarol,35,Chicago"
    )
    csv_bytes = csv_content.encode("utf-8")
    base64_string = base64.b64encode(csv_bytes).decode("utf-8")

    llm = ChatOpenAI(model=MODEL_NAME, use_responses_api=True)

    # Test LangChain standard format
    langchain_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How many people are in this CSV file?",
            },
            {
                "type": "file",
                "base64": base64_string,
                "mime_type": "text/csv",
                "filename": "people.csv",
            },
        ],
    }
    payload = llm._get_request_payload([langchain_message])
    block = payload["input"][0]["content"][1]
    assert block["type"] == "input_file"

    response = llm.invoke([langchain_message])
    assert isinstance(response, AIMessage)
    assert response.content
    assert (
        "3" in str(response.content).lower() or "three" in str(response.content).lower()
    )

    # Test OpenAI native format
    openai_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How many people are in this CSV file?",
            },
            {
                "type": "input_file",
                "filename": "people.csv",
                "file_data": f"data:text/csv;base64,{base64_string}",
            },
        ],
    }
    payload2 = llm._get_request_payload([openai_message])
    block2 = payload2["input"][0]["content"][1]
    assert block2["type"] == "input_file"

    response2 = llm.invoke([openai_message])
    assert isinstance(response2, AIMessage)
    assert response2.content
    assert (
        "3" in str(response2.content).lower()
        or "three" in str(response2.content).lower()
    )


@pytest.mark.default_cassette("test_phase.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_phase(output_version: str) -> None:
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        verbosity="high",
        reasoning={"effort": "medium", "summary": "auto"},
        output_version=output_version,
    )

    agent = create_agent(model, tools=[get_weather])

    input_message = {
        "role": "user",
        "content": (
            "What's the weather in the oldest major city in the US? State your answer "
            "and then generate a tool call this turn."
        ),
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    text_block = next(
        block for block in first_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "commentary"

    final_response = result["messages"][-1]
    text_block = next(
        block for block in final_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "final_answer"


@pytest.mark.default_cassette("test_phase_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_phase_streaming(output_version: str) -> None:
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        verbosity="high",
        reasoning={"effort": "medium", "summary": "auto"},
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=[get_weather])

    input_message = {
        "role": "user",
        "content": (
            "What's the weather in the oldest major city in the US? State your answer "
            "and then generate a tool call this turn."
        ),
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    if output_version == "responses/v1":
        assert [block["type"] for block in first_response.content] == [
            "reasoning",
            "text",
            "function_call",
        ]
    else:
        assert [block["type"] for block in first_response.content] == [
            "reasoning",
            "text",
            "tool_call",
        ]
    text_block = next(
        block for block in first_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "commentary"

    final_response = result["messages"][-1]
    assert [block["type"] for block in final_response.content] == ["text"]
    text_block = next(
        block for block in final_response.content if block["type"] == "text"
    )
    assert text_block["phase"] == "final_answer"


@pytest.mark.default_cassette("test_tool_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_tool_search(output_version: str) -> None:
    @tool(extras={"defer_loading": True})
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    @tool(extras={"defer_loading": True})
    def get_recipe(query: str) -> None:
        """Get a recipe for chicken soup."""

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        output_version=output_version,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_recipe, {"type": "tool_search"}],
    )
    input_message = {"role": "user", "content": "What's the weather in San Francisco?"}
    result = agent.invoke({"messages": [input_message]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls
    if output_version == "v1":
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "tool_call",
        ]
    else:
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "tool_search_call",
            "tool_search_output",
            "function_call",
        ]

    assert isinstance(result["messages"][2], ToolMessage)

    assert result["messages"][3].text


@pytest.mark.default_cassette("test_tool_search_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["responses/v1", "v1"])
def test_tool_search_streaming(output_version: str) -> None:
    @tool(extras={"defer_loading": True})
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    @tool(extras={"defer_loading": True})
    def get_recipe(query: str) -> None:
        """Get a recipe for chicken soup."""

    model = ChatOpenAI(
        model="gpt-5.4",
        use_responses_api=True,
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, get_recipe, {"type": "tool_search"}],
    )
    input_message = {"role": "user", "content": "What's the weather in San Francisco?"}
    result = agent.invoke({"messages": [input_message]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    assert isinstance(tool_call_message, AIMessage)
    assert tool_call_message.tool_calls
    if output_version == "v1":
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "server_tool_call",
            "server_tool_result",
            "tool_call",
        ]
    else:
        assert [block["type"] for block in tool_call_message.content] == [  # type: ignore[index]
            "tool_search_call",
            "tool_search_output",
            "function_call",
        ]

    assert isinstance(result["messages"][2], ToolMessage)

    assert result["messages"][3].text


@pytest.mark.vcr
def test_client_executed_tool_search() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    def search_tools(goal: str) -> list[dict]:
        """Search for available tools to help answer the question."""
        return [
            {
                "type": "function",
                "defer_loading": True,
                **convert_to_openai_tool(get_weather)["function"],
            }
        ]

    tool_search_schema = convert_to_openai_tool(search_tools, strict=True)
    tool_search_config: dict = {
        "type": "tool_search",
        "execution": "client",
        "description": tool_search_schema["function"]["description"],
        "parameters": tool_search_schema["function"]["parameters"],
    }

    class ClientToolSearchMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["model"])
        def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return None
            for block in last_message.content:
                if isinstance(block, dict) and block.get("type") == "tool_search_call":
                    call_id = block.get("call_id")
                    args = block.get("arguments", {})
                    goal = args.get("goal", "") if isinstance(args, dict) else ""
                    loaded_tools = search_tools(goal)
                    tool_search_output = {
                        "type": "tool_search_output",
                        "execution": "client",
                        "call_id": call_id,
                        "status": "completed",
                        "tools": loaded_tools,
                    }
                    return {
                        "messages": [HumanMessage(content=[tool_search_output])],
                        "jump_to": "model",
                    }
            return None

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Any,
        ) -> Any:
            if request.tool_call["name"] == "get_weather":
                return handler(request.override(tool=get_weather))
            return handler(request)

    llm = ChatOpenAI(model="gpt-5.4", use_responses_api=True)

    agent = create_agent(
        model=llm,
        tools=[tool_search_config],
        middleware=[ClientToolSearchMiddleware()],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather in San Francisco?")]}
    )
    messages = result["messages"]
    search_tool_call = messages[1]
    assert search_tool_call.content[0]["type"] == "tool_search_call"

    search_tool_output = messages[2]
    assert search_tool_output.content[0]["type"] == "tool_search_output"

    tool_call = messages[3]
    assert tool_call.tool_calls

    assert isinstance(messages[4], ToolMessage)

    assert messages[5].text
