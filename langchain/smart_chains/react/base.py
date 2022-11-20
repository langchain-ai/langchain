"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, Optional, Tuple

from pydantic import BaseModel

from langchain.chains.llm import LLMChain
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain.smart_chains.react.prompt import PROMPT
from langchain.smart_chains.router import LLMRouterChain
from langchain.smart_chains.router_expert import ExpertConfig, RouterExpertChain


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
        """Name of the action of when to finish the chain."""
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
    """Class to assist with exploration of a document store."""

    def __init__(self, docstore: Docstore):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = docstore
        self.document: Optional[Document] = None

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = self.docstore.search(term)
        if isinstance(result, Document):
            self.document = result
            return self.document.summary
        else:
            self.document = None
            return result

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        return self.document.lookup(term)


class ReActChain(RouterExpertChain):
    """Chain that implements the ReAct paper.

    Example:
        .. code-block:: python

            from langchain import ReActChain, OpenAI
            react = ReAct(llm=OpenAI())
    """

    def __init__(self, llm: LLM, docstore: Docstore, **kwargs: Any):
        """Initialize with the LLM and a docstore."""
        router_chain = ReActRouterChain(llm)
        docstore_explorer = DocstoreExplorer(docstore)
        expert_configs = [
            ExpertConfig(expert_name="Search", expert=docstore_explorer.search),
            ExpertConfig(expert_name="Lookup", expert=docstore_explorer.lookup),
        ]
        super().__init__(
            router_chain=router_chain, expert_configs=expert_configs, **kwargs
        )
