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
from langchain.chains.router_expert import RouterExpertChain, ExpertConfig


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

    @property
    def finish_action_name(self) -> str:
        """The action name of when to finish the chain."""
        return "Finish"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return f"Observation {self.i - 1}: "

    @property
    def router_prefix(self) -> str:
        """Prefix to append the router call with."""
        return f"Thought {self.i}:"


class DocstoreExplorer:

    def __init__(self, docstore: Docstore):
        self.docstore=docstore
        self.document = None

    def search(self, term: str):
        result = self.docstore.search(term)
        if isinstance(result, Document):
            self.document = result
            return self.document.summary
        else:
            self.document = None
            return result

    def lookup(self, term: str):
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        return self.document.lookup(term)


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
        docstore_explorer = DocstoreExplorer(self.docstore)
        expert_configs = [
            ExpertConfig(expert_name="Search", expert=docstore_explorer.search),
            ExpertConfig(expert_name="Lookup", expert=docstore_explorer.lookup)
        ]
        chain = RouterExpertChain(
            router_chain=router_chain,
            expert_configs=expert_configs,
            verbose=self.verbose
        )
        output = chain.run(question)
        return {self.output_key: output}
