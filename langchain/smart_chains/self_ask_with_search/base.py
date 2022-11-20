"""Chain that does self ask with search."""
from typing import Any, Tuple

from langchain.chains.llm import LLMChain
from langchain.chains.serpapi import SerpAPIChain
from langchain.llms.base import LLM
from langchain.smart_chains.router import LLMRouterChain
from langchain.smart_chains.router_expert import ExpertConfig, RouterExpertChain
from langchain.smart_chains.self_ask_with_search.prompt import PROMPT


class SelfAskWithSearchRouter(LLMRouterChain):
    """Router for the self-ask-with-search paper."""

    def __init__(self, llm: LLM, **kwargs: Any):
        """Initialize with an LLM."""
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        super().__init__(llm_chain=llm_chain, **kwargs)

    def _extract_action_and_input(self, text: str) -> Tuple[str, str]:
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


class SelfAskWithSearchChain(RouterExpertChain):
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
        expert_configs = [
            ExpertConfig(expert_name="Intermediate Answer", expert=search_chain.run)
        ]
        super().__init__(
            router_chain=router,
            expert_configs=expert_configs,
            starter_string="\nAre follow up questions needed here:",
            **kwargs
        )
