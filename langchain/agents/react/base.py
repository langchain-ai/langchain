"""Chain that implements the ReAct paper from https://arxiv.org/pdf/2210.03629.pdf."""
import re
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from langchain.agents.agent import Agent, AgentExecutor
from langchain.agents.react.textworld_prompt import TEXTWORLD_PROMPT
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.tools import Tool
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.tools.base import BaseTool


class ReActDocstoreAgent(Agent, BaseModel):
    """Agent for the ReAct chain."""

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "react-docstore"

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        return WIKI_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        if len(tools) != 2:
            raise ValueError(f"Exactly two tools must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Lookup", "Search"}:
            raise ValueError(
                f"Tool names should be Lookup and Search, got {tool_names}"
            )

    def _fix_text(self, text: str) -> str:
        return text + "\nAction:"

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        action_prefix = "Action: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            return None
        action_block = text.strip().split("\n")[-1]

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
        return "Observation: "

    @property
    def _stop(self) -> List[str]:
        return ["\nObservation:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"


class DocstoreExplorer:
    """Class to assist with exploration of a document store."""

    def __init__(self, docstore: Docstore):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = docstore
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = self.docstore.search(term)
        if isinstance(result, Document):
            self.document = result
            return self._summary
        else:
            self.document = None
            return result

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _summary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> List[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")


class ReActTextWorldAgent(ReActDocstoreAgent, BaseModel):
    """Agent for the ReAct TextWorld chain."""

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        return TEXTWORLD_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
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
