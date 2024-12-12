"""Test ChatNaver chat model."""

from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_community.chat_models import ChatClovaX


def test_stream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    for token in llm.stream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    async for token in llm.astream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.abatch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.abatch(["I'm Clova", "I'm not Clova"], config={"tags": ["foo"]})
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = llm.batch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = await llm.ainvoke("I'm Clova", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX()

    result = llm.invoke("I'm Clova", config=dict(tags=["foo"]))
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
