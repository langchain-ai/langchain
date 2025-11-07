"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool, Tool
from pydantic import Field
from typing_extensions import override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.react.output_parser import ReActOutputParser
from langchain_classic.agents.react.textworld_prompt import TEXTWORLD_PROMPT
from langchain_classic.agents.react.wiki_prompt import WIKI_PROMPT
from langchain_classic.agents.utils import validate_tools_single_input

if TYPE_CHECKING:
    from langchain_community.docstore.base import Docstore


_LOOKUP_AND_SEARCH_TOOLS = {"Lookup", "Search"}


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class ReActDocstoreAgent(Agent):
    """Agent for the ReAct chain."""

    output_parser: AgentOutputParser = Field(default_factory=ReActOutputParser)

    @classmethod
    @override
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return AgentType.REACT_DOCSTORE

    @classmethod
    @override
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        return WIKI_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != len(_LOOKUP_AND_SEARCH_TOOLS):
            msg = f"Exactly two tools must be specified, but got {tools}"
            raise ValueError(msg)
        tool_names = {tool.name for tool in tools}
        if tool_names != _LOOKUP_AND_SEARCH_TOOLS:
            msg = f"Tool names should be Lookup and Search, got {tool_names}"
            raise ValueError(msg)

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def _stop(self) -> list[str]:
        return ["\nObservation:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class DocstoreExplorer:
    """Class to assist with exploration of a document store."""

    def __init__(self, docstore: Docstore):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = docstore
        self.document: Document | None = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = self.docstore.search(term)
        if isinstance(result, Document):
            self.document = result
            return self._summary
        self.document = None
        return result

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            msg = "Cannot lookup without a successful search first"
            raise ValueError(msg)
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        if self.lookup_index >= len(lookups):
            return "No More Results"
        result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
        return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _summary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            msg = "Cannot get paragraphs without a document"
            raise ValueError(msg)
        return self.document.page_content.split("\n\n")


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class ReActTextWorldAgent(ReActDocstoreAgent):
    """Agent for the ReAct TextWorld chain."""

    @classmethod
    @override
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        return TEXTWORLD_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != 1:
            msg = f"Exactly one tool must be specified, but got {tools}"
            raise ValueError(msg)
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Play"}:
            msg = f"Tool name should be Play, got {tool_names}"
            raise ValueError(msg)


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class ReActChain(AgentExecutor):
    """[Deprecated] Chain that implements the ReAct paper."""

    def __init__(self, llm: BaseLanguageModel, docstore: Docstore, **kwargs: Any):
        """Initialize with the LLM and a docstore."""
        docstore_explorer = DocstoreExplorer(docstore)
        tools = [
            Tool(
                name="Search",
                func=docstore_explorer.search,
                description="Search for a term in the docstore.",
            ),
            Tool(
                name="Lookup",
                func=docstore_explorer.lookup,
                description="Lookup a term in the docstore.",
            ),
        ]
        agent = ReActDocstoreAgent.from_llm_and_tools(llm, tools)
        super().__init__(agent=agent, tools=tools, **kwargs)
