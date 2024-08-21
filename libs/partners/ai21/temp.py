from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_ai21.chat_models import ChatAI21
from langchain_core.utils.function_calling import convert_to_openai_tool
from libs.core.langchain_core.globals import set_verbose

set_verbose(True)

@tool
def get_weather(place: str, date: str) -> str:
    """Return the weather at place in the given date."""
    if place == "New York" and date == "2024-12-05":
        return "25 celsius"
    elif place == "New York" and date == "2024-12-06":
        return "27 celsius"
    elif place == "London" and date == "2024-12-05":
        return "22 celsius"
    return "32 celsius"

llm = ChatAI21(model="jamba-1.5-mini",
               max_tokens=2000,
               temperature=0,
               api_host="https://api.ai21.com",
               api_key="API_KEY")

llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])

chat_messages = [SystemMessage(content="You are a helpful assistant")]

human_messages = [
    HumanMessage(content="What is the forecast for the weather in New York on December 5, 2024?"),
    HumanMessage(content="And what about the 2024-12-06?"),
    HumanMessage(content="OK, thank you."),
    HumanMessage(content="And what is the expected weather in London on December 5, 2024?")]


for human_message in human_messages:
    print(f"User: {human_message.content}")
    chat_messages.append(human_message)
    response = llm_with_tools.invoke(chat_messages)
    chat_messages.append(response)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "get_weather":
            weather = get_weather.invoke(
                {"place": tool_call["args"]["place"], "date": tool_call["args"]["date"]})
            chat_messages.append(ToolMessage(content=weather, tool_call_id=tool_call["id"]))
            llm_answer = llm_with_tools.invoke(chat_messages)
            print(f"Assistant: {llm_answer.content}")
    else:
        print(f"Assistant: {response.content}")
