"""Test ChatGroq chat model."""

import json
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_groq import ChatGroq
from tests.unit_tests.fake.callbacks import (
    FakeCallbackHandler,
    FakeCallbackHandlerWithChatStart,
)


#
# Smoke test Runnable interface
#
@pytest.mark.scheduled
def test_invoke() -> None:
    """Test Chat wrapper."""
    chat = ChatGroq(
        temperature=0.7,
        base_url=None,
        groq_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
async def test_ainvoke() -> None:
    """Test ainvoke tokens from ChatGroq."""
    chat = ChatGroq(max_tokens=10)

    result = await chat.ainvoke("Welcome to the Groqetship!", config={"tags": ["foo"]})
    assert isinstance(result, BaseMessage)
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_batch() -> None:
    """Test batch tokens from ChatGroq."""
    chat = ChatGroq(max_tokens=10)

    result = chat.batch(["Hello!", "Welcome to the Groqetship!"])
    for token in result:
        assert isinstance(token, BaseMessage)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_abatch() -> None:
    """Test abatch tokens from ChatGroq."""
    chat = ChatGroq(max_tokens=10)

    result = await chat.abatch(["Hello!", "Welcome to the Groqetship!"])
    for token in result:
        assert isinstance(token, BaseMessage)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_stream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(max_tokens=10)

    for token in chat.stream("Welcome to the Groqetship!"):
        assert isinstance(token, BaseMessageChunk)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_astream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(max_tokens=10)

    async for token in chat.astream("Welcome to the Groqetship!"):
        assert isinstance(token, BaseMessageChunk)
        assert isinstance(token.content, str)


#
# Test Legacy generate methods
#
@pytest.mark.scheduled
def test_generate() -> None:
    """Test sync generate."""
    n = 1
    chat = ChatGroq(max_tokens=10)
    message = HumanMessage(content="Hello", n=1)
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_agenerate() -> None:
    """Test async generation."""
    n = 1
    chat = ChatGroq(max_tokens=10, n=1)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


#
# Test streaming flags in invoke and generate
#
@pytest.mark.scheduled
def test_invoke_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatGroq(
        max_tokens=2,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
async def test_agenerate_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandlerWithChatStart()
    chat = ChatGroq(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output is not None
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


#
# Misc tests
#
def test_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    chat = ChatGroq(
        max_tokens=2,
        temperature=0,
        callbacks=[callback],
    )
    list(chat.stream("Respond with the single word Hello", stop=["o"]))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert isinstance(generation, LLMResult)
    assert generation.generations[0][0].text == "Hell"


def test_system_message() -> None:
    """Test ChatGroq wrapper with system message."""
    chat = ChatGroq(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq()

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

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


@pytest.mark.scheduled
def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""
    llm = ChatGroq()

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice=True)

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


def test_streaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq()

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.stream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


async def test_astreaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq()

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.astream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    async for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


@pytest.mark.scheduled
def test_json_mode_structured_output() -> None:
    """Test with_structured_output with json"""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    chat = ChatGroq().with_structured_output(Joke, method="json_mode")
    result = chat.invoke(
        "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
    )
    assert type(result) == Joke
    assert len(result.setup) != 0
    assert len(result.punchline) != 0


# Groq does not currently support N > 1
# @pytest.mark.scheduled
# def test_chat_multiple_completions() -> None:
#     """Test ChatGroq wrapper with multiple completions."""
#     chat = ChatGroq(max_tokens=10, n=5)
#     message = HumanMessage(content="Hello")
#     response = chat._generate([message])
#     assert isinstance(response, ChatResult)
#     assert len(response.generations) == 5
#     for generation in response.generations:
#          assert isinstance(generation.message, BaseMessage)
#          assert isinstance(generation.message.content, str)
