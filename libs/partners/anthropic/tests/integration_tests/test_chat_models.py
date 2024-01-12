"""Test ChatAnthropicMessages chat model."""
from langchain_core.prompts import ChatPromptTemplate

from langchain_anthropic.chat_models import ChatAnthropicMessages


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
