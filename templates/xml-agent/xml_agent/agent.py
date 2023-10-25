from langchain.chat_models import ChatAnthropic
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents.format_scratchpad import format_xml
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.retrievers.you import YouRetriever
from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
from langchain.pydantic_v1 import BaseModel


model = ChatAnthropic(model="claude-2")

# Fake Tool
retriever = YouRetriever(k=5)
retriever_tool = create_retriever_tool(retriever, "search", "Use this to search for current events.")

tools = [retriever_tool]

prompt = hub.pull("hwchase17/xml-agent")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = model.bind(stop=["</tool_input>"])
agent = {
    "question": lambda x: x["question"],
    "agent_scratchpad": lambda x: format_xml(x['intermediate_steps']),
} | prompt | llm_with_stop | XMLAgentOutputParser()

class AgentInput(BaseModel):
    question: str

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)

agent_executor = agent_executor | (lambda x: x["output"])
