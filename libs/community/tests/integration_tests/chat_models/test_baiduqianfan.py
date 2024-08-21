from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

from langchain_community.chat_models import QianfanChatEndpoint


@tool
def get_current_weather(location: str, unit: str = "摄氏度") -> str:
    """获取指定地点的天气"""
    return f"{location}是晴朗，25{unit}左右。"


def test_chat_qianfan_tool_result_to_model() -> None:
    """Test QianfanChatEndpoint invoke with tool_calling result."""
    messages = [
        HumanMessage("上海天气怎么样？"),
        AIMessage(
            content=" ",
            tool_calls=[
                ToolCall(
                    name="get_current_weather",
                    args={"location": "上海", "unit": "摄氏度"},
                    id="foo",
                    type="tool_call",
                ),
            ],
        ),
        ToolMessage(
            content="上海是晴天，25度左右。",
            tool_call_id="foo",
            name="get_current_weather",
        ),
    ]
    chat = QianfanChatEndpoint(model="ERNIE-3.5-8K")  # type: ignore[call-arg]
    llm_with_tool = chat.bind_tools([get_current_weather])
    response = llm_with_tool.invoke(messages)
    assert isinstance(response, AIMessage)
    print(response.content)  # noqa: T201
