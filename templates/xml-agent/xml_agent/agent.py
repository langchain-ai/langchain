from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_xml
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.render import render_text_description
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from xml_agent.prompts import conversational_prompt, parse_output


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


model = ChatAnthropic(model="claude-3-sonnet-20240229")

tools = [DuckDuckGoSearchRun()]

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
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})


agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
).with_types(input_type=AgentInput)

agent_executor = agent_executor | (lambda x: x["output"])
