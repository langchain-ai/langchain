"""Test Integration chat model."""
from __module_name__.chat_models import ChatIntegration


def test_integration_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatIntegration()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_integration_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatIntegration()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_integration_abatch() -> None:
    """Test streaming tokens from ChatIntegration."""
    llm = ChatIntegration()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_integration_abatch_tags() -> None:
    """Test batch tokens from ChatIntegration."""
    llm = ChatIntegration()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_integration_batch() -> None:
    """Test batch tokens from ChatIntegration."""
    llm = ChatIntegration()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_integration_ainvoke() -> None:
    """Test invoke tokens from ChatIntegration."""
    llm = ChatIntegration()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_integration_invoke() -> None:
    """Test invoke tokens from ChatIntegration."""
    llm = ChatIntegration()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
