"""
Tests the agent created by langchain_cohere.create_cohere_react_agent

You will need to set:
* COHERE_API_KEY
* TAVILY_API_KEY
"""

from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from langchain_cohere import ChatCohere, create_cohere_react_agent


def test_invoke():
    llm = ChatCohere()

    internet_search = TavilySearchResults(max_results=4)
    internet_search.name = "internet_search"
    internet_search.description = "Route a user query to the internet"

    prompt = ChatPromptTemplate.from_template("{input}")

    agent = create_cohere_react_agent(llm, [internet_search], prompt)

    agent_executor = AgentExecutor(agent=agent, tools=[internet_search])

    actual = agent_executor.invoke(
        {
            "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",  # noqa: E501
        }
    )
    expected = "Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999."  # noqa: E501

    assert expected == actual
