"""Chain that does self ask with search."""
from typing import Any, Tuple

from langchain.chains.llm import LLMChain
from langchain.chains.serpapi import SerpAPIChain
from langchain.llms.base import LLM
from langchain.routing_chains.router import LLMRouter
from langchain.routing_chains.routing_chain import RoutingChain, ToolConfig
from langchain.routing_chains.self_ask_with_search.prompt import PROMPT


class SelfAskWithSearchRouter(LLMRouter):
    """Router for the self-ask-with-search paper."""

    def __init__(self, llm: LLM, **kwargs: Any):
        """Initialize with an LLM."""
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        super().__init__(llm_chain=llm_chain, **kwargs)

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
    def router_prefix(self) -> str:
        """Prefix to append the router call with."""
        return ""

    @property
    def starter_string(self) -> str:
        """Put this string after user input but before first router call."""
        return "\nAre follow up questions needed here:"


class SelfAskWithSearchChain(RoutingChain):
    """Chain that does self ask with search.

    Example:
        .. code-block:: python

            from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain
            search_chain = SerpAPIChain()
            self_ask = SelfAskWithSearchChain(llm=OpenAI(), search_chain=search_chain)
    """

    def __init__(self, llm: LLM, search_chain: SerpAPIChain, **kwargs: Any):
        """Initialize with just an LLM and a search chain."""
        intermediate = "\nIntermediate answer:"
        router = SelfAskWithSearchRouter(llm, stops=[intermediate])
        search_tool = ToolConfig(tool_name="Intermediate Answer", tool=search_chain.run)
        expert_configs = [search_tool]
        super().__init__(router=router, expert_configs=expert_configs, **kwargs)
