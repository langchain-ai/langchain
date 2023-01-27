"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from langchain.agents.agent import Agent, AgentExecutor
from langchain.agents.react.textworld_prompt import TEXTWORLD_PROMPT
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.tools import Tool
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


class ReActDocstoreAgent(Agent, BaseModel):
    """Agent for the ReAct chain."""

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "react-docstore"

    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        """Return default prompt."""
        return WIKI_PROMPT

    i: int = 1

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        if len(tools) != 2:
            raise ValueError(f"Exactly two tools must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Lookup", "Search"}:
            raise ValueError(
                f"Tool names should be Lookup and Search, got {tool_names}"
            )

    def _prepare_for_new_call(self) -> None:
        self.i = 1

    def _fix_text(self, text: str) -> str:
        return text + f"\nAction {self.i}:"

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        action_prefix = f"Action {self.i}: "
        if not text.split("\n")[-1].startswith(action_prefix):
            return None
        self.i += 1
        action_block = text.split("\n")[-1]

        action_str = action_block[len(action_prefix) :]
        # Parse out the action and the directive.
        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise ValueError(f"Could not parse action directive: {action_str}")
        return re_matches.group(1), re_matches.group(2)

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool of when to finish the chain."""
        return "Finish"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return f"Observation {self.i - 1}: "

    @property
    def _stop(self) -> List[str]:
        return [f"\nObservation {self.i}:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
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


class ReActTextWorldAgent(ReActDocstoreAgent, BaseModel):
    """Agent for the ReAct TextWorld chain."""

    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        """Return default prompt."""
        return TEXTWORLD_PROMPT

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        if len(tools) != 1:
            raise ValueError(f"Exactly one tool must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Play"}:
            raise ValueError(f"Tool name should be Play, got {tool_names}")


class ReActChain(AgentExecutor):
    """Chain that implements the ReAct paper.

    Example:
        .. code-block:: python

            from langchain import ReActChain, OpenAI
            react = ReAct(llm=OpenAI())
    """

    def __init__(self, llm: BaseLLM, docstore: Docstore, **kwargs: Any):
        """Initialize with the LLM and a docstore."""
        docstore_explorer = DocstoreExplorer(docstore)
        tools = [
            Tool(name="Search", func=docstore_explorer.search),
            Tool(name="Lookup", func=docstore_explorer.lookup),
        ]
        agent = ReActDocstoreAgent.from_llm_and_tools(llm, tools)
        super().__init__(agent=agent, tools=tools, **kwargs)
