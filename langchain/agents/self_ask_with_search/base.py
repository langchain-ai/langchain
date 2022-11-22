"""Chain that does self ask with search."""
from typing import Any, List, Tuple

from langchain.chains.llm import LLMChain
from langchain.chains.serpapi import SerpAPIChain
from langchain.llms.base import LLM
from langchain.agents.agent import Agent
from langchain.agents.self_ask_with_search.prompt import PROMPT
from langchain.agents.tools import Tool


class SelfAskWithSearchAgent(Agent):
    """Agent for the self-ask-with-search paper."""

    @classmethod
    def from_llm_and_tools(
        cls, llm: LLM, tools: List[Tool], **kwargs: Any
    ) -> "SelfAskWithSearchAgent":
        """Construct an agent from an LLM and tools."""
        if len(tools) != 1:
            raise ValueError(f"Exactly one tool must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Intermediate Answer"}:
            raise ValueError(
                f"Tool name should be Intermediate Answer, got {tool_names}"
            )

        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        return cls(llm_chain=llm_chain, tools=tools, **kwargs)

    def _extract_tool_and_input(self, text: str) -> Tuple[str, str]:
        followup = "Follow up:"
        if "\n" not in text:
            last_line = text
        else:
            last_line = text.split("\n")[-1]

        if followup not in last_line:
            finish_string = "So the final answer is: "
            if finish_string not in last_line:
                raise ValueError("We should probably never get here")
            return "Final Answer", text[len(finish_string) :]

        if ":" not in last_line:
            after_colon = last_line
        else:
            after_colon = text.split(":")[-1]

        if " " == after_colon[0]:
            after_colon = after_colon[1:]
        if "?" != after_colon[-1]:
            print("we probably should never get here..." + text)

        return "Intermediate Answer", after_colon

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Intermediate answer: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return ""

    @property
    def starter_string(self) -> str:
        """Put this string after user input but before first LLM call."""
        return "\nAre follow up questions needed here:"


class SelfAskWithSearchChain(SelfAskWithSearchAgent):
    """Chain that does self ask with search.

    Example:
        .. code-block:: python

            from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain
            search_chain = SerpAPIChain()
            self_ask = SelfAskWithSearchChain(llm=OpenAI(), search_chain=search_chain)
    """

    def __init__(self, llm: LLM, search_chain: SerpAPIChain, **kwargs: Any):
        """Initialize with just an LLM and a search chain."""
        search_tool = Tool(name="Intermediate Answer", func=search_chain.run)
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        super().__init__(llm_chain=llm_chain, tools=[search_tool], **kwargs)
