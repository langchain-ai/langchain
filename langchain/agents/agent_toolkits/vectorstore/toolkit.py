"""Toolkit for interacting with a vector store."""
from typing import List, Sequence

from pydantic import BaseModel, Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.tools import BaseTool
from langchain.tools.vectorstore.tool import (
    VectorStoreQATool,
    VectorStoreQAWithSourcesTool,
)
from langchain.vectorstores.base import VectorStore


class NamedVectorStore(BaseModel):
    """A vector store with a name."""

    name: str
    vectorstore: VectorStore


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a vector store."""

    vectorstores: Sequence[NamedVectorStore] = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        for named_vectorstore in self.vectorstores:
            tools.append(
                VectorStoreQATool(
                    name=named_vectorstore.name,
                    vectorstore=named_vectorstore.vectorstore,
                    llm=self.llm,
                )
            )
            tools.append(
                VectorStoreQAWithSourcesTool(
                    name=named_vectorstore.name,
                    vectorstore=named_vectorstore.vectorstore,
                    llm=self.llm,
                )
            )
        return tools
