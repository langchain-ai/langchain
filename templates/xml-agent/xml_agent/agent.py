from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.agents.format_scratchpad import format_xml
from langchain.chat_models import ChatAnthropic
from langchain.pydantic_v1 import BaseModel
from langchain.retrievers.you import YouRetriever
from langchain.schema import AIMessage, HumanMessage
from langchain.tools.render import render_text_description

from xml_agent.prompts import conversational_prompt, parse_output


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


model = ChatAnthropic(model="claude-2")

# Fake Tool
retriever = YouRetriever(k=5)
retriever_tool = create_retriever_tool(
    retriever, "search", "Use this to search for current events."
)

tools = [retriever_tool]

prompt = conversational_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = model.bind(stop=["</tool_input>"])

agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_xml(x["intermediate_steps"]),
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    }
    | prompt
    | llm_with_stop
    | parse_output
)


class AgentInput(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]]


agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
).with_types(input_type=AgentInput)

agent_executor = agent_executor | (lambda x: x["output"])
