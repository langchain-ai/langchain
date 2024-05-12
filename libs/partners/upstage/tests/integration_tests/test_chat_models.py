import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_upstage import ChatUpstage


def test_chat_upstage_model() -> None:
    """Test ChatUpstage wrapper handles model_name."""
    chat = ChatUpstage(model="foo")
    assert chat.model_name == "foo"
    chat = ChatUpstage(model_name="bar")
    assert chat.model_name == "bar"


def test_chat_upstage_system_message() -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatUpstage(max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_upstage_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatUpstage(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_upstage_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatUpstage(max_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_upstage_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatUpstage(
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_chat_upstage_extra_kwargs() -> None:
    """Test extra kwargs to chat upstage."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatUpstage(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatUpstage(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatUpstage(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatUpstage(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatUpstage(model_kwargs={"model": "solar-1-mini-chat"})


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatUpstage()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatUpstage()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatUpstage."""
    llm = ChatUpstage()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatUpstage."""
    llm = ChatUpstage()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatUpstage."""
    llm = ChatUpstage()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatUpstage."""
    llm = ChatUpstage()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatUpstage."""
    llm = ChatUpstage()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
