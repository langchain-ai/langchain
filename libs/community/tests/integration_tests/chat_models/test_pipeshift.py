import pytest  # type: ignore[import-not-found]
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.pipeshift import ChatPipeshift

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def test_chat_pipeshift_model() -> None:
    """Test ChatPipeshift wrapper handles model."""
    chat = ChatPipeshift(model="foo")  # type: ignore[call-arg]
    assert chat.model == "foo"
    chat = ChatPipeshift(model="bar")  # type: ignore[call-arg]
    assert chat.model == "bar"


@pytest.mark.enable_socket
def test_chat_pipeshift_system_message() -> None:
    """Test ChatPipeshift wrapper with system message."""
    chat = ChatPipeshift(max_tokens=10)  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    messages = [system_message, human_message]
    response = chat.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


# @pytest.mark.enable_socket
# def test_chat_pipeshift_llm_output_contains_model_name() -> None:
#     """Test llm_output contains model."""
#     chat = ChatPipeshift(max_tokens=10)
#     message = HumanMessage(content="Hello")
#     llm_result = chat.generate([[message]])
#     assert llm_result.llm_output is not None
#     assert llm_result.llm_output["model"] == chat.model

# @pytest.mark.enable_socket
# def test_chat_pipeshift_streaming_llm_output_contains_model_name() -> None:
#     """Test llm_output contains model."""
#     chat = ChatPipeshift(max_tokens=10, streaming=True)
#     message = HumanMessage(content="Hello")
#     llm_result = chat.generate([[message]])
#     assert llm_result.llm_output is not None
#     assert llm_result.llm_output["model"] == chat.model

# @pytest.mark.enable_socket
# def test_chat_pipeshift_invalid_streaming_params() -> None:
#     """Test that streaming correctly invokes on_llm_new_token callback."""
#     with pytest.raises(ValueError):
#         ChatPipeshift(
#             max_tokens=10,
#             streaming=True,
#             temperature=0,
#             n=5,
#         )


@pytest.mark.enable_socket
def test_chat_pipeshift_extra_kwargs() -> None:
    """Test extra kwargs to chat pipeshift."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatPipeshift(foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatPipeshift(foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatPipeshift(foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatPipeshift(model_kwargs={"temperature": 0.2})  # type: ignore[call-arg]

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatPipeshift(model_kwargs={"model": DEFAULT_MODEL})  # type: ignore[call-arg]


@pytest.mark.enable_socket
def test_stream() -> None:
    """Test streaming tokens from Pipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.enable_socket
async def test_astream() -> None:
    """Test streaming tokens from Pipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.enable_socket
async def test_abatch() -> None:
    """Test streaming tokens from ChatPipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.enable_socket
async def test_abatch_tags() -> None:
    """Test batch tokens from ChatPipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.enable_socket
def test_batch() -> None:
    """Test batch tokens from ChatPipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.enable_socket
async def test_ainvoke() -> None:
    """Test invoke tokens from ChatPipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.enable_socket
def test_invoke() -> None:
    """Test invoke tokens from ChatPipeshift."""
    llm = ChatPipeshift()  # type: ignore[call-arg]

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
