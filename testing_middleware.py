from langchain.agents.new_agent import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import operator
from dataclasses import dataclass
from typing import Annotated
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy

from langchain.agents.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate

class State(AgentState):
    model_request_count: Annotated[int, operator.add]

class ModelRequestLimitMiddleware(AgentMiddleware):
    """Terminates after N model requests"""

    state_schema = State

    def __init__(self, max_requests: int = 10):
        self.max_requests = max_requests

    def before_model(self, state: State) -> AgentUpdate | AgentJump | None:
        # TODO: want to be able to configure end behavior here
        if state.get("model_request_count", 0) == self.max_requests:
            return {"jump_to": "__end__"}

        return {"model_request_count": 1}



@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city"""

    return f"The weather in {city} is sunny."


class WeatherResponse(BaseModel):
    city: str
    weather: str


# state extension (note we only make 3 tool calls below)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather],
    system_prompt="You are a weather agent. You are tasked with fetching the weather for a given city.",
    middleware=[ModelRequestLimitMiddleware(max_requests=3)],
)
agent = agent.compile()

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Please check the weather in SF, NYC, LA, and Boston.")
        ]
    }
)
for msg in result["messages"]:
    msg.pretty_print()

"""
================================ Human Message =================================

Please check the weather in SF, NYC, LA, and Boston.
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_7LddqyVgqxjTYm84UUfFBFZA)
 Call ID: call_7LddqyVgqxjTYm84UUfFBFZA
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is sunny.
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_gUL7CHn6YqE80M9M5G5miA3k)
 Call ID: call_gUL7CHn6YqE80M9M5G5miA3k
  Args:
    city: New York City
================================= Tool Message =================================
Name: get_weather

The weather in New York City is sunny.
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_asOAXRkPbBWBdt4SzQGPYQab)
 Call ID: call_asOAXRkPbBWBdt4SzQGPYQab
  Args:
    city: Los Angeles
================================= Tool Message =================================
Name: get_weather

The weather in Los Angeles is sunny.
"""

# structured response


agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather],
    system_prompt="You are a weather agent. You are tasked with fetching the weather for a given city.",
    middleware=[ModelRequestLimitMiddleware(max_requests=3)],
    response_format=ToolStrategy(WeatherResponse),
)
agent = agent.compile()

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Please check the weather in SF")
        ]
    }
)

print(repr(result["response"]))
#> WeatherResponse(city='SF', weather='sunny')


# builtin provider tool support (web search for ex)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[{"type": "web_search_preview"}],
    system_prompt="You are a weather agent. You are tasked with fetching the weather for a given city. Please use the web search tool to fetch the weather.",
    # response_format=WeatherResponse,
)
agent = agent.compile()

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is the weather in SF?")
        ]
    }
)
for msg in result["messages"]:
    msg.pretty_print()

"""
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================

[{'type': 'text', 'text': 'As of 1:58 PM PDT on Friday, September 5, 2025, the weather in San Francisco, CA, is mostly cloudy with a temperature of 66°F (19°C). ([weather.com](https://weather.com/weather/today/l/San%2BFrancisco%2BCA?canonicalCityId=e7784799733d2133bcb75674a102b347&utm_source=openai))\n\n## Weather for San Francisco, CA:\nCurrent Conditions: Cloudy, 58°F (14°C)\n\nDaily Forecast:\n* Friday, September 5: Low: 60°F (15°C), High: 69°F (20°C), Description: Low clouds breaking for some sun\n* Saturday, September 6: Low: 61°F (16°C), High: 69°F (21°C), Description: Areas of low clouds, then sun and pleasant\n* Sunday, September 7: Low: 63°F (17°C), High: 72°F (22°C), Description: Areas of low clouds, then sun and pleasant\n* Monday, September 8: Low: 63°F (17°C), High: 71°F (21°C), Description: Low clouds breaking for some sun\n* Tuesday, September 9: Low: 60°F (16°C), High: 70°F (21°C), Description: Morning low clouds followed by clouds giving way to some sun\n* Wednesday, September 10: Low: 56°F (13°C), High: 68°F (20°C), Description: Mostly cloudy with a shower in places\n* Thursday, September 11: Low: 56°F (13°C), High: 69°F (21°C), Description: Partly sunny\n ', 'annotations': [{'end_index': 274, 'start_index': 134, 'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com', 'type': 'url_citation', 'url': 'https://weather.com/weather/today/l/San%2BFrancisco%2BCA?canonicalCityId=e7784799733d2133bcb75674a102b347&utm_source=openai'}]}]
"""

# system prompt and tools as None

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=None,
    system_prompt=None,
    middleware=[ModelRequestLimitMiddleware(max_requests=3)],
)
agent = agent.compile()

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is 2 + 2?")
        ]
    }
)
result["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

2 + 2 equals 4.
"""

# a call and call model



