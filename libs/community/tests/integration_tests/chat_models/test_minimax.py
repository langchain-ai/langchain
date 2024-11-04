import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_community.chat_models import MiniMaxChat


def test_chat_minimax_not_group_id() -> None:
    if "MINIMAX_GROUP_ID" in os.environ:
        del os.environ["MINIMAX_GROUP_ID"]
    chat = MiniMaxChat()  # type: ignore[call-arg]
    response = chat.invoke("你好呀")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_minimax_with_stream() -> None:
    chat = MiniMaxChat()  # type: ignore[call-arg]
    for chunk in chat.stream("你好呀"):
        assert isinstance(chunk, AIMessage)
        assert isinstance(chunk.content, str)


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


def test_chat_minimax_with_tool() -> None:
    """Test MinimaxChat with bind tools."""
    chat = MiniMaxChat()  # type: ignore[call-arg]
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


class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str
    justification: str


def test_chat_minimax_with_structured_output() -> None:
    """Test MiniMaxChat with structured output."""
    llm = MiniMaxChat()  # type: ignore
    structured_llm = llm.with_structured_output(AnswerWithJustification)
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, AnswerWithJustification)


def test_chat_tongyi_with_structured_output_include_raw() -> None:
    """Test MiniMaxChat with structured output."""
    llm = MiniMaxChat()  # type: ignore
    structured_llm = llm.with_structured_output(
        AnswerWithJustification, include_raw=True
    )
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, dict)
    assert isinstance(response.get("raw"), AIMessage)
    assert isinstance(response.get("parsed"), AnswerWithJustification)
