"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.react.prompt import PROMPT
from langchain.chains.router import LLMRouterChain
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.input import ChainedInput
from langchain.llms.base import LLM


class ReActRouterChain(LLMRouterChain, BaseModel):
    """Router for the ReAct chin."""

    i: int = 1

    def __init__(self, llm: LLM, **kwargs: Any):
        """Initialize with the language model."""
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        stops = ["\nObservation 1:"]
        super().__init__(llm_chain=llm_chain, stops=stops, **kwargs)

    def _fix_text(self, text: str) -> str:
        return text + f"\nAction {self.i}:"

    def _extract_action_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        action_prefix = f"Action {self.i}: "
        if not text.split("\n")[-1].startswith(action_prefix):
            return None
        self.i += 1
        self.stops = [f"\nObservation {self.i}:"]
        action_block = text.split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise ValueError(f"Could not parse action directive: {action_str}")
        return re_matches.group(1), re_matches.group(2)


class ReActChain(Chain, BaseModel):
    """Chain that implements the ReAct paper.

    Example:
        .. code-block:: python

            from langchain import ReActChain, OpenAI
            react = ReAct(llm=OpenAI())
    """

    llm: LLM
    """LLM wrapper to use."""
    docstore: Docstore
    """Docstore to use."""
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
        question = inputs[self.input_key]
        router_chain = ReActRouterChain(self.llm)
        chained_input = ChainedInput(f"{question}\nThought 1:", verbose=self.verbose)
        document = None
        while True:
            action, directive, ret_text = router_chain.get_action_and_input(
                chained_input.input
            )
            chained_input.add(ret_text, color="green")
            if action == "Search":
                result = self.docstore.search(directive)
                if isinstance(result, Document):
                    document = result
                    observation = document.summary
                else:
                    document = None
                    observation = result
            elif action == "Lookup":
                if document is None:
                    raise ValueError("Cannot lookup without a successful search first")
                observation = document.lookup(directive)
            elif action == "Finish":
                return {self.output_key: directive}
            else:
                raise ValueError(f"Got unknown action directive: {action}")
            chained_input.add(f"\nObservation {router_chain.i - 1}: ")
            chained_input.add(observation, color="yellow")
            chained_input.add(f"\nThought {router_chain.i}:")
