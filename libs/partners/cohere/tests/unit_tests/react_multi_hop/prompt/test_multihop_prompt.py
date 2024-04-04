from typing import Any, Dict, List, Tuple, Type

import pytest
from freezegun import freeze_time
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_cohere.react_multi_hop.prompt import multi_hop_prompt
from tests.unit_tests.react_multi_hop import ExpectationType, read_expectation_from_file


class InternetSearchTool(BaseTool):
    class _InputSchema(BaseModel):
        query: str = Field(type=str, description="Query to search the internet with")

    name = "internet_search"
    description = (
        "Returns a list of relevant document snippets for a textual query "
        "retrieved from the internet"
    )
    args_schema: Type[_InputSchema] = _InputSchema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


TOOLS: List[BaseTool] = [InternetSearchTool()]  # type: ignore
DOCUMENTS = [
    {
        "URL": "https://www.cnbc.com/2015/05/26/19-famous-companies-that-originally-had-different-names.html",
        "title": "19 famous companies that originally had different names",
        "text": 'Sound of Music made more money during this "best buy" four-day sale than it did in a typical month – thus, the store was renamed to Best Buy in 1983.\n4. Apple Computers » Apple, Inc.\nFounded in 1976, the tech giant we know today as Apple was originally named Apple Computers by founders Steve Jobs, Ronald Wayne and Steve Wozniak. In 2007, Jobs announced that the company was dropping the word "Computer" from its name to better reflect their move into a wider field of consumer electronics. "The Mac, iPod, Apple TV and iPhone. Only one of those is a computer.',  # noqa: E501
    },
    {
        "URL": "https://en.wikipedia.org/wiki/The_Sound_of_Music_(film)",
        "title": "The Sound of Music (film) - Wikipedia",
        "text": 'In 1966, American Express created the first Sound of Music guided tour in Salzburg. Since 1972, Panorama Tours has been the leading Sound of Music bus tour company in the city, taking approximately 50,000 tourists a year to various film locations in Salzburg and the surrounding region. Although the Salzburg tourism industry took advantage of the attention from foreign tourists, residents of the city were apathetic about "everything that is dubious about tourism." The guides on the bus tour "seem to have little idea of what really happened on the set." Even the ticket agent for the Sound of Music Dinner Show tried to dissuade Austrians from attending a performance that was intended for American tourists, saying that it "does not have anything to do with the real Austria."',  # noqa: E501
    },
    {
        "URL": "https://en.wikipedia.org/wiki/Best_Buy",
        "title": "Best Buy - Wikipedia",
        "text": "Concept IV stores included an open layout with products organized by category, cash registers located throughout the store, and slightly smaller stores than Concept III stores. The stores also had large areas for demonstrating home theater systems and computer software.\nIn 1999, Best Buy was added to Standard & Poor's S&P 500.\n2000s\nIn 2000, Best Buy formed Redline Entertainment, an independent music label and action-sports video distributor. The company acquired Magnolia Hi-Fi, Inc., an audio-video retailer located in California, Washington, and Oregon, in December 2000.\nIn January 2001, Best Buy acquired Musicland Stores Corporation, a Minnetonka, Minnesota-based retailer that sold home-entertainment products under the Sam Goody, Suncoast Motion Picture Company, Media Play, and OnCue brands.",  # noqa: E501
    },
    {
        "URL": "https://en.wikipedia.org/wiki/Best_Buy",
        "title": "Best Buy - Wikipedia",
        "text": 'Later that year, Best Buy opened its first superstore in Burnsville, Minnesota. The Burnsville location featured a high-volume, low-price business model, which was borrowed partially from Schulze\'s successful Tornado Sale in 1981. In its first year, the Burnsville store out-performed all other Best Buy stores combined.\nBest Buy was taken public in 1985, and two years later it debuted on the New York Stock Exchange. In 1988, Best Buy was in a price and location war with Detroit-based appliance chain Highland Superstores, and Schulze attempted to sell the company to Circuit City for US$30 million. Circuit City rejected the offer, claiming they could open a store in Minneapolis and "blow them away."',  # noqa: E501
    },
]
COMPLETIONS = [
    """Plan: First, I need to find out which company was originally called Sound of Music, then I need to find out when it was added to the S&P 500.
Action: ```json
[
    {
        "tool_name": "internet_search",
        "parameters": {
            "query": "which company was originally called sound of music"
        }
    }
]
```""",  # noqa: E501
    """Reflection: I found out that Sound of Music was renamed Best Buy in 1983, now I need to find out when Best Buy was added to the S&P 500.
Action: ```json
[
    {
        "tool_name": "internet_search",
        "parameters": {
            "query": "when was best buy added to S&P 500"
        }
    }
]
```""",  # noqa: E501,
]
MESSAGES = [
    HumanMessage(content="Hello, how are you doing?"),
    AIMessage(content="I'm doing well, thanks!"),
    HumanMessage(
        content="In what year was the company that was founded as Sound of Music added to the S&P 500?"  # noqa: E501
    ),
]


@freeze_time("Saturday, March 30, 2024 13:20:40")
@pytest.mark.parametrize(
    "tools,template,invoke_with,intermediate_steps,scenario_name",
    [
        pytest.param(
            [TOOLS[0]],
            ChatPromptTemplate.from_template("{input}"),
            {
                "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?"  # noqa: E501
            },
            [],
            "base",
            id="base",
        ),
        pytest.param(
            [TOOLS[0]],
            ChatPromptTemplate.from_template("{input}"),
            {
                "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?"  # noqa: E501
            },
            [
                (
                    AgentActionMessageLog(
                        tool=TOOLS[0].name,
                        tool_input={
                            "query": "which company was originally called sound of music"  # noqa: E501
                        },
                        log="\nFirst I will search for the company founded as Sound of Music. Then I will search for the year this company was added to the S&P 500.{'tool_name': 'internet_search', 'parameters': {'query': 'company founded as Sound of Music'}}\n",  # noqa: E501
                        message_log=[AIMessage(content=COMPLETIONS[0])],
                    ),
                    [DOCUMENTS[0], DOCUMENTS[1]],
                ),
            ],
            "base_after_one_hop",
            id="after one hop",
        ),
        pytest.param(
            [TOOLS[0]],
            ChatPromptTemplate.from_template("{input}"),
            {
                "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?"  # noqa: E501
            },
            [
                (
                    AgentActionMessageLog(
                        tool=TOOLS[0].name,
                        tool_input={
                            "query": "which company was originally called sound of music"  # noqa: E501
                        },
                        log="\nFirst I will search for the company founded as Sound of Music. Then I will search for the year this company was added to the S&P 500.{'tool_name': 'internet_search', 'parameters': {'query': 'company founded as Sound of Music'}}\n",  # noqa: E501
                        message_log=[AIMessage(content=COMPLETIONS[0])],
                    ),
                    [DOCUMENTS[0], DOCUMENTS[1]],
                ),
                (
                    AgentActionMessageLog(
                        tool=TOOLS[0].name,
                        tool_input={"query": "when was best buy added to S&P 500"},
                        log="\nI found out that Sound of Music was renamed Best Buy in 1983, now I need to find out when Best Buy was added to the S&P 500.\n{'tool_name': 'internet_search', 'parameters': {'query': 'when was best buy added to S&P 500'}}\n",  # noqa: E501
                        message_log=[AIMessage(content=COMPLETIONS[1])],
                    ),
                    [DOCUMENTS[2], DOCUMENTS[3]],
                ),
            ],
            "base_after_two_hops",
            id="after two hops",
        ),
        pytest.param(
            [TOOLS[0]],
            ChatPromptTemplate.from_messages([MESSAGES[0], MESSAGES[1], MESSAGES[2]]),
            {},
            [],
            "base_with_chat_history",
            id="base with chat history",
        ),
    ],
)
def test_multihop_prompt(
    tools: List[BaseTool],
    template: ChatPromptTemplate,
    invoke_with: Dict[str, Any],
    intermediate_steps: List[Tuple[AgentAction, Any]],
    scenario_name: str,
) -> None:
    """Tests prompt rendering against hardcoded expectations."""
    expected = read_expectation_from_file(ExpectationType.prompts, scenario_name)
    chain = RunnablePassthrough.assign(
        agent_scratchpad=lambda _: [],  # Usually provided by create_cohere_react_agent.
        intermediate_steps=lambda _: intermediate_steps,
    ) | multi_hop_prompt(tools=tools, prompt=template)

    actual = chain.invoke(invoke_with)  # type: StringPromptValue  # type: ignore

    assert StringPromptValue == type(actual)
    assert expected == actual.text
