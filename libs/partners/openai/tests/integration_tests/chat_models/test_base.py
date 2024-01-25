"""Test ChatOpenAI chat model."""
from typing import Any, Optional

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import ChatOpenAI
from tests.unit_tests.fake.callbacks import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_openai() -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(
        temperature=0.7,
        base_url=None,
        organization=None,
        openai_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_openai_model() -> None:
    """Test ChatOpenAI wrapper handles model_name."""
    chat = ChatOpenAI(model="foo")
    assert chat.model_name == "foo"
    chat = ChatOpenAI(model_name="bar")
    assert chat.model_name == "bar"


def test_chat_openai_system_message() -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_openai_generate() -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatOpenAI(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_openai_multiple_completions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_openai_streaming_generation_info() -> None:
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
    callback_manager = CallbackManager([callback])
    chat = ChatOpenAI(
        max_tokens=2,
        temperature=0,
        callback_manager=callback_manager,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatOpenAI(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


@pytest.mark.scheduled
async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = ChatOpenAI(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_openai_bind_functions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""

    class Person(BaseModel):
        """Identifying information about a person."""

        name: str = Field(..., title="Name", description="The person's name")
        age: int = Field(..., title="Age", description="The person's age")
        fav_food: Optional[str] = Field(
            default=None, title="Fav Food", description="The person's favorite food"
        )

    chat = ChatOpenAI(
        max_tokens=30,
        n=1,
        streaming=True,
    ).bind_functions(functions=[Person], function_call="Person")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use the provided Person function"),
            ("user", "{input}"),
        ]
    )

    chain = prompt | chat

    message = HumanMessage(content="Sally is 13 years old")
    response = await chain.abatch([{"input": message}])

    assert isinstance(response, list)
    assert len(response) == 1
    for generation in response:
        assert isinstance(generation, AIMessage)


def test_chat_openai_extra_kwargs() -> None:
    """Test extra kwargs to chat openai."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatOpenAI(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatOpenAI(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatOpenAI(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatOpenAI(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatOpenAI(model_kwargs={"model": "gpt-3.5-turbo-instruct"})


@pytest.mark.scheduled
def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_abatch() -> None:
    """Test streaming tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_ainvoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_openai_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_logprobs() -> None:
    llm = ChatOpenAI()
    result = llm.generate([[HumanMessage(content="I'm PickleRick")]], logprobs=True)
    assert result.generations[0][0].generation_info
    assert "content" in result.generations[0][0].generation_info["logprobs"]


async def test_async_logprobs() -> None:
    llm = ChatOpenAI()
    result = await llm.agenerate(
        [[HumanMessage(content="I'm PickleRick")]], logprobs=True
    )
    assert result.generations[0][0].generation_info
    assert "content" in result.generations[0][0].generation_info["logprobs"]


def test_logprobs_streaming() -> None:
    llm = ChatOpenAI()
    result = llm.generate(
        [[HumanMessage(content="I'm PickleRick")]], logprobs=True, stream=True
    )
    assert result.generations[0][0].generation_info
    assert "content" in result.generations[0][0].generation_info["logprobs"]


async def test_async_logprobs_streaming() -> None:
    llm = ChatOpenAI()
    result = await llm.agenerate(
        [[HumanMessage(content="I'm PickleRick")]], logprobs=True, stream=True
    )
    assert result.generations[0][0].generation_info
    assert "content" in result.generations[0][0].generation_info["logprobs"]
