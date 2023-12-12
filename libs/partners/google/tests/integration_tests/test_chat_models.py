"""Test ChatGoogleGenerativeAI chat model."""
from langchain_google.chat_models import ChatGoogleGenerativeAI


def test_integration_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_integration_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_integration_abatch() -> None:
    """Test streaming tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_integration_abatch_tags() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_integration_batch() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_integration_ainvoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_integration_invoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model="gemini-nano")

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
