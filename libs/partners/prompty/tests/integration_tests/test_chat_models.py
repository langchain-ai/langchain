"""Test ChatPrompty chat model."""
from langchain_prompty.chat_models import ChatPrompty


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatPrompty()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatPrompty()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatPrompty."""
    llm = ChatPrompty()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatPrompty."""
    llm = ChatPrompty()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatPrompty."""
    llm = ChatPrompty()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatPrompty."""
    llm = ChatPrompty()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatPrompty."""
    llm = ChatPrompty()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
