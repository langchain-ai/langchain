"""Toolkit for interacting with a vector store."""
from typing import List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.tools import BaseTool
from langchain.tools.vectorstore.tool import (
    VectorStoreQATool,
    VectorStoreQAWithSourcesTool,
)
from langchain.vectorstores.base import VectorStore


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a vector store."""

    vectorstore: VectorStore = Field(exclude=True)
    name: str
    description: str
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        qa_tool = VectorStoreQATool(
            name=self.name,
            extra_description=self.description,
            vectorstore=self.vectorstore,
            llm=self.llm,
        )
        qa_with_sources_tool = VectorStoreQAWithSourcesTool(
            name=f"{self.name}_with_sources",
            extra_description=self.description,
            vectorstore=self.vectorstore,
            llm=self.llm,
        )
        return [qa_tool, qa_with_sources_tool]
