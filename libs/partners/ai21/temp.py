from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from langchain_ai21.chat_models import ChatAI21
from langchain_core.utils.function_calling import convert_to_openai_tool
from libs.core.langchain_core.globals import set_verbose

set_verbose(True)

@tool
def get_weather(place: str, date: str) -> str:
    """Return the weather at place in the given date."""
    print(f"Fetching the expected weather at {place} during {date} from the internet...")
    return "32 celsius"

llm = ChatAI21(model="jamba-1.5-large",
               max_tokens=500,
               api_host="https://api.ai21.com",
               api_key="...")

llm_with_tools = llm.bind_tools([convert_to_openai_tool(get_weather)])

messages = [SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="What is the expected weather in New York on 2024-12-05?")]

response = llm_with_tools.invoke(messages)
messages.append(response)
print(response.tool_calls)
tool_call = response.tool_calls[0]

if tool_call["name"] == "get_weather":
    place = tool_call["args"]["place"]
    date = tool_call["args"]["date"]
    weather = get_weather.invoke({"place": place, "date": date})
    messages.append(ToolMessage(content=weather, tool_call_id=tool_call["id"]))
    final_answer = llm_with_tools.invoke(messages)
    print(final_answer)