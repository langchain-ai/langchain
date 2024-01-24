"""Test ChatNVIDIA chat model."""
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


def test_chat_ai_endpoints() -> None:
    """Test ChatNVIDIA wrapper."""
    chat = ChatNVIDIA(
        model="llama2_13b",
        temperature=0.7,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_ai_endpoints_model() -> None:
    """Test wrapper handles model."""
    chat = ChatNVIDIA(model="mistral")
    assert chat.model == "mistral"


def test_chat_ai_endpoints_system_message() -> None:
    """Test wrapper with system message."""
    chat = ChatNVIDIA(model="llama2_13b", max_tokens=36)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


## TODO: Not sure if we want to support the n syntax. Trash or keep test


def test_ai_endpoints_streaming() -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=36)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_ai_endpoints_astream() -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=35)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_ai_endpoints_abatch() -> None:
    """Test streaming tokens."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=36)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_abatch_tags() -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=55)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_ai_endpoints_batch() -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=60)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_ainvoke() -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=60)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_ai_endpoints_invoke() -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model="llama2_13b", max_tokens=60)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
