"""
Tests the agent created by langchain_cohere.create_cohere_react_agent

You will need to set:
* COHERE_API_KEY
"""
from typing import Any, Type

from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from langchain_cohere import ChatCohere, create_cohere_react_agent


def test_invoke_multihop_agent() -> None:
    llm = ChatCohere(temperature=0.0)

    class _InputSchema(BaseModel):
        query: str = Field(description="Query to search the internet with")

    class InternetSearchTool(BaseTool):
        """Mimic an internet search engine"""

        name: str = "internet_search"
        description: str = "Returns a list of relevant document snippets for a textual query retrieved from the internet"  # noqa: E501
        args_schema: Type[BaseModel] = _InputSchema

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            query = kwargs.get("query", "")
            if "sound of music" in query.lower():
                return [
                    {
                        "URL": "https://www.cnbc.com/2015/05/26/19-famous-companies-that-originally-had-different-names.html",  # noqa: E501
                        "title": "19 famous companies that originally had different names",  # noqa: E501
                        "text": 'Sound of Music made more money during this "best buy" four-day sale than it did in a typical month – thus, the store was renamed to Best Buy in 1983.\n4. Apple Computers » Apple, Inc.\nFounded in 1976, the tech giant we know today as Apple was originally named Apple Computers by founders Steve Jobs, Ronald Wayne and Steve Wozniak. In 2007, Jobs announced that the company was dropping the word "Computer" from its name to better reflect their move into a wider field of consumer electronics. "The Mac, iPod, Apple TV and iPhone. Only one of those is a computer.',  # noqa: E501
                    },
                    {
                        "URL": "https://en.wikipedia.org/wiki/The_Sound_of_Music_(film)",
                        "title": "The Sound of Music (film) - Wikipedia",
                        "text": 'In 1966, American Express created the first Sound of Music guided tour in Salzburg. Since 1972, Panorama Tours has been the leading Sound of Music bus tour company in the city, taking approximately 50,000 tourists a year to various film locations in Salzburg and the surrounding region. Although the Salzburg tourism industry took advantage of the attention from foreign tourists, residents of the city were apathetic about "everything that is dubious about tourism." The guides on the bus tour "seem to have little idea of what really happened on the set." Even the ticket agent for the Sound of Music Dinner Show tried to dissuade Austrians from attending a performance that was intended for American tourists, saying that it "does not have anything to do with the real Austria."',  # noqa: E501
                    },
                ]
            elif "best buy" in query.lower():
                return [
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

            return []

    tools = [InternetSearchTool()]
    prompt = ChatPromptTemplate.from_template("{input}")

    agent = create_cohere_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools)

    actual = agent_executor.invoke(
        {
            "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",  # noqa: E501
        }
    )

    assert "output" in actual
    assert "best buy" in actual["output"].lower()
    assert "citations" in actual
