from langchain.agents import tool
from langchain.chat_models import ChatAnthropic
model = ChatAnthropic(model="claude-2")
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain.agents.format_scratchpad import format_xml
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults

# Fake Tool
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

tools = [tavily_tool]

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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
