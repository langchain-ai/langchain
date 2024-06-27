"""Test ChatNaver chat model."""
from langchain_naver.chat_models import ChatNaver


def test_stream() -> None:
    """Test streaming tokens from ChatNaver."""
    llm = ChatNaver()

    for token in llm.stream("I'm Clova"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatNaver."""
    llm = ChatNaver()

    async for token in llm.astream("I'm Clova"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatNaver."""
    llm = ChatNaver()

    result = await llm.abatch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatNaver."""
    llm = ChatNaver()

    result = await llm.abatch(
        ["I'm Clova", "I'm not Clova"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatNaver."""
    llm = ChatNaver()

    result = llm.batch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatNaver."""
    llm = ChatNaver()

    result = await llm.ainvoke("I'm Clova", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatNaver."""
    llm = ChatNaver()

    result = llm.invoke("I'm Clova", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
