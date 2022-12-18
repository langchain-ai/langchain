"""Chain that does self ask with search."""
from typing import Any, List, Optional, Tuple

from langchain.agents.agent import Agent, AgentWithTools
from langchain.agents.self_ask_with_search.prompt import PROMPT
from langchain.agents.tools import Tool
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate
from langchain.serpapi import SerpAPIWrapper


class SelfAskWithSearchAgent(Agent):
    """Agent for the self-ask-with-search paper."""

    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        """Prompt does not depend on tools."""
        return PROMPT

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        if len(tools) != 1:
            raise ValueError(f"Exactly one tool must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Intermediate Answer"}:
            raise ValueError(
                f"Tool name should be Intermediate Answer, got {tool_names}"
            )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        followup = "Follow up:"
        last_line = text.split("\n")[-1]

        if followup not in last_line:
            finish_string = "So the final answer is: "
            if finish_string not in last_line:
                return None
            return "Final Answer", last_line[len(finish_string) :]

        after_colon = text.split(":")[-1]

        if " " == after_colon[0]:
            after_colon = after_colon[1:]

        return "Intermediate Answer", after_colon

    def _fix_text(self, text: str) -> str:
        return f"{text}\nSo the final answer is:"

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
        return "Are follow up questions needed here:"


class SelfAskWithSearchChain(AgentWithTools):
    """Chain that does self ask with search.

    Example:
        .. code-block:: python

            from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIWrapper
            search_chain = SerpAPIWrapper()
            self_ask = SelfAskWithSearchChain(llm=OpenAI(), search_chain=search_chain)
    """

    def __init__(self, llm: LLM, search_chain: SerpAPIWrapper, **kwargs: Any):
        """Initialize with just an LLM and a search chain."""
        search_tool = Tool(name="Intermediate Answer", func=search_chain.run)
        agent = SelfAskWithSearchAgent.from_llm_and_tools(llm, [search_tool])
        super().__init__(agent=agent, tools=[search_tool], **kwargs)
