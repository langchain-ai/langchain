"""Test ZhipuAI Chat Model."""

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool

from langchain_community.chat_models.zhipuai import ChatZhipuAI
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_default_call() -> None:
    """Test default model call."""
    chat = ChatZhipuAI()
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = ChatZhipuAI(model="glm-4")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatZhipuAI()

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = ChatZhipuAI(streaming=True)
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatZhipuAI()
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


def test_tool_call() -> None:
    """Test tool calling by ChatZhipuAI"""
    chat = ChatZhipuAI(model="glm-4-long")  # type: ignore[call-arg]
    tools = [add, multiply]
    chat_with_tools = chat.bind_tools(tools)

    query = "What is 3 * 12?"
    messages = [HumanMessage(query)]
    ai_msg = chat_with_tools.invoke(messages)
    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call
    messages.append(ai_msg)  # type: ignore[arg-type]
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])  # type: ignore[attr-defined]
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))  # type: ignore[arg-type]
    response = chat_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)
