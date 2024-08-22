import os
from getpass import getpass
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ai21.chat_models import ChatAI21
from langchain_core.utils.function_calling import convert_to_openai_tool

os.environ["AI21_API_KEY"] = getpass()

@tool
def get_weather(location: str, date: str) -> str:
    """“Provide the weather for the specified location on the given date.”"""
    if location == "New York" and date == "2024-12-05":
        return "25 celsius"
    elif location == "New York" and date == "2024-12-06":
        return "27 celsius"
    elif location == "London" and date == "2024-12-05":
        return "22 celsius"
    return "32 celsius"

llm = ChatAI21(model="jamba-1.5-mini")

llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])

chat_messages = [SystemMessage(content="You are a helpful assistant. You can use the provided tools "
                                       "to assist with various tasks and provide accurate information")]

human_messages = [
    HumanMessage(content="What is the forecast for the weather in New York on December 5, 2024?"),
    HumanMessage(content="And what about the 2024-12-06?"),
    HumanMessage(content="OK, thank you."),
    HumanMessage(content="What is the expected weather in London on December 5, 2024?")]


for human_message in human_messages:
    print(f"User: {human_message.content}")
    chat_messages.append(human_message)
    response = llm_with_tools.invoke(chat_messages)
    chat_messages.append(response)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "get_weather":
            weather = get_weather.invoke(
                {"location": tool_call["args"]["location"], "date": tool_call["args"]["date"]})
            chat_messages.append(ToolMessage(content=weather, tool_call_id=tool_call["id"]))
            llm_answer = llm_with_tools.invoke(chat_messages)
            print(f"Assistant: {llm_answer.content}")
    else:
        print(f"Assistant: {response.content}")