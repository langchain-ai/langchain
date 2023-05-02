from langchain.agents.chat.base import ChatAgent
from langchain.llms.openai import OpenAI
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun


class TestAgent:
    def test_agent_generation(self) -> None:
        web_search = DuckDuckGoSearchRun()
        tools = [web_search]
        agent = ChatAgent.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=tools,
            llm=OpenAI(maxTokens=10),
        )
        assert agent.allowed_tools == set([web_search.name])
