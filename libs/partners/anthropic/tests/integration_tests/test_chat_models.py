"""Test ChatAnthropic chat model."""

from typing import List

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate

from langchain_anthropic import ChatAnthropic, ChatAnthropicMessages
from tests.unit_tests._utils import FakeCallbackHandler


def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_system_invoke() -> None:
    """Test invoke tokens with a system message"""
    llm = ChatAnthropicMessages(model_name="claude-instant-1.2")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert cartographer. If asked, you are a cartographer. "
                "STAY IN CHARACTER",
            ),
            ("human", "Are you a mathematician?"),
        ]
    )

    chain = prompt | llm

    result = chain.invoke({})
    assert isinstance(result.content, str)


def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    chat = ChatAnthropic(model="test")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_anthropic_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatAnthropic(model="test")
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatAnthropic(model="test")
    message = HumanMessage(content="Hello")
    response = chat.stream([message])
    for token in response:
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)


def test_anthropic_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model="test",
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    for token in chat.stream([message]):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
    assert callback_handler.llm_streams > 1


async def test_anthropic_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model="test",
        callback_manager=callback_manager,
        verbose=True,
    )
    chat_messages: List[BaseMessage] = [
        HumanMessage(content="How many toes do dogs have?")
    ]
    async for token in chat.astream(chat_messages):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
    assert callback_handler.llm_streams > 1
