from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

# Fake Tool
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

tools = [tavily_tool]

llm = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]]


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)

agent_executor = agent_executor | (lambda x: x["output"])
