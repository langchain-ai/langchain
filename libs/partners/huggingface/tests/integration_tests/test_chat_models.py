"""Test Hugging Face Inference Providers API wrapper

You will need HF_TOKEN set in your environment to run these tests.
"""

import json
from typing import Optional

from langchain_huggingface.llms import HuggingFaceEndpoint

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from pydantic import BaseModel

from langchain_huggingface.chat_models import ChatHuggingFace


def test_chat_hf_call() -> None:
    """Test valid call to HF Inference Providers."""
    llm = HuggingFaceEndpoint(
    endpoint_url="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    provider="novita",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )

    resp = chat_hf.invoke("Hello!")
    assert isinstance(resp, AIMessage)

    assert len(resp.content) > 0


def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    task="conversational",
    provider="together",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = chat_hf.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"
    assert isinstance(resp.tool_calls, list)
    assert len(resp.tool_calls) == 1
    tool_call = resp.tool_calls[0]
    assert tool_call["name"] == "MyTool"
    assert tool_call["args"] == {"age": 27, "name": "Erick"}


def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    task="conversational",
    provider="together",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )
    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = chat_hf.bind_tools([MyTool], tool_choice=True)

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


def test_stream() -> None:
    """Test streaming tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    endpoint_url="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    provider="novita",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )

    for token in chat_hf.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    provider="fireworks-ai",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )
    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in chat_hf.astream("I'm Pickle Rick"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
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
    """Test abatch tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    endpoint_url="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    provider="novita",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )
    result = await chat_hf.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)




def test_batch() -> None:
    """Test batch tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    endpoint_url="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    provider="novita",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )
    result = chat_hf.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )

    result = await chat_hf.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatHuggingFace."""
    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    )
    chat_hf = ChatHuggingFace(  # type: ignore[call-arg]
        llm=llm, temperature=0, max_tokens=100
    )
    result = chat_hf.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


