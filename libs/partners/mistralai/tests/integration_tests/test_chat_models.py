"""Test ChatMistral chat model."""
from langchain_core.messages import HumanMessage

from langchain_mistralai.chat_models import ChatMistralAI


def test_stream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatMistralAI"""
    llm = ChatMistralAI()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatMistralAI"""
    llm = ChatMistralAI()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatMistralAI"""
    llm = ChatMistralAI()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatMistralAI"""
    llm = ChatMistralAI()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatMistralAI"""
    llm = ChatMistralAI()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_chat_mistralai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatMistralAI(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


def test_chat_mistralai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatMistralAI(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model

