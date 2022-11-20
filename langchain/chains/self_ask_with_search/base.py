"""Chain that does self ask with search."""
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.router import LLMRouterChain
from langchain.chains.self_ask_with_search.prompt import PROMPT
from langchain.chains.serpapi import SerpAPIChain
from langchain.input import ChainedInput
from langchain.llms.base import LLM


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
            return "Final Answer", text

        if ":" not in last_line:
            after_colon = last_line
        else:
            after_colon = text.split(":")[-1]

        if " " == after_colon[0]:
            after_colon = after_colon[1:]
        if "?" != after_colon[-1]:
            print("we probably should never get here..." + text)

        return "Intermediate Answer", after_colon


class SelfAskWithSearchChain(Chain, BaseModel):
    """Chain that does self ask with search.

    Example:
        .. code-block:: python

            from langchain import SelfAskWithSearchChain, OpenAI, SerpAPIChain
            search_chain = SerpAPIChain()
            self_ask = SelfAskWithSearchChain(llm=OpenAI(), search_chain=search_chain)
    """

    llm: LLM
    """LLM wrapper to use."""
    search_chain: SerpAPIChain
    """Search chain to use."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        chained_input = ChainedInput(inputs[self.input_key], verbose=self.verbose)
        chained_input.add("\nAre follow up questions needed here:")
        intermediate = "\nIntermediate answer:"
        router = SelfAskWithSearchRouter(self.llm, stops=[intermediate])
        action, action_input, log = router.get_action_and_input(chained_input.input)
        chained_input.add(log, color="green")
        while action != "Final Answer":
            external_answer = self.search_chain.run(action_input)
            chained_input.add(intermediate + " ")
            chained_input.add(external_answer + ".", color="yellow")
            action, action_input, log = router.get_action_and_input(chained_input.input)
            chained_input.add(log, color="green")
        return {self.output_key: action_input}
