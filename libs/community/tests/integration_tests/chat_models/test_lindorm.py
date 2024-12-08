"""Test Lindorm AI Chat Model."""

import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.lindorm_chat import ChatLindormAI


class Config:
    AI_CHAT_LLM_ENDPOINT = os.environ.get("AI_CHAT_LLM_ENDPOINT", "<CHAT_ENDPOINT>")
    AI_CHAT_USERNAME = os.environ.get("AI_CHAT_USERNAME", "root")
    AI_CHAT_PWD = os.environ.get("AI_CHAT_PWD", "<PASSWORD>")
    AI_DEFAULT_CHAT_MODEL = "qa_model_qwen_72b_chat"


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatLindormAI(
            model_name=Config.AI_DEFAULT_CHAT_MODEL,
            endpoint=Config.AI_CHAT_LLM_ENDPOINT,
            username=Config.AI_CHAT_USERNAME,
            password=Config.AI_CHAT_PWD,
            client=None,
        ),
    ]:
        assert model.model_name == Config.AI_DEFAULT_CHAT_MODEL


def test_default_call() -> None:
    """Test default model call."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD,
        client=None,
    )  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    # print("response: ", response)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD,
        client=None,
    )  # type: ignore[call-arg]
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    # print("response: ", response)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatLindormAI(
        model_name=Config.AI_DEFAULT_CHAT_MODEL,
        endpoint=Config.AI_CHAT_LLM_ENDPOINT,
        username=Config.AI_CHAT_USERNAME,
        password=Config.AI_CHAT_PWD,
    )  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        # print("generations: ", generations)
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
