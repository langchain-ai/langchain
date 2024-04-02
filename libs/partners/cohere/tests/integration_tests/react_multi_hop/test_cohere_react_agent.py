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


def test_invoke_multihop_agent() -> None:
    llm = ChatCohere(temperature=0.0)

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

    accepted_outputs = [
        "Best Buy, the company founded as Sound of Music, was added to the S&P 500 in 1999.",  # noqa: E501
        "Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999.",  # noqa: E501
        "Best Buy, the company founded as Sound of Music, was added to the S&P 500 in 1999. The company was renamed Best Buy in 1983, when it became Best Buy Company, Inc.",  # noqa: E501
        "Sorry, I could not find any information about the company founded as Sound of Music being added to the S&P 500. However, I did find that Best Buy, the company founded as Sound of Music in 1966, was added to the S&P index in 1985, two years after its debut on the New York Stock Exchange.",  # noqa: E501
    ]

    assert "output" in actual
    assert actual["output"] in accepted_outputs
