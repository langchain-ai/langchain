from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_xml
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.render import render_text_description
from langchain_community.llms import OpenAI
from langchain_core.pydantic_v1 import BaseModel

from solo_performance_prompting_agent.parser import parse_output
from solo_performance_prompting_agent.prompts import conversational_prompt

_model = OpenAI()
_tools = [DuckDuckGoSearchRun()]
_prompt = conversational_prompt.partial(
    tools=render_text_description(_tools),
    tool_names=", ".join([t.name for t in _tools]),
)
_llm_with_stop = _model.bind(stop=["</tool_input>", "</final_answer>"])

agent = (
    {
        "question": lambda x: x["question"],
        "agent_scratchpad": lambda x: format_xml(x["intermediate_steps"]),
    }
    | _prompt
    | _llm_with_stop
    | parse_output
)


class AgentInput(BaseModel):
    question: str


agent_executor = AgentExecutor(
    agent=agent, tools=_tools, verbose=True, handle_parsing_errors=True
).with_types(input_type=AgentInput)

agent_executor = agent_executor | (lambda x: x["output"])
