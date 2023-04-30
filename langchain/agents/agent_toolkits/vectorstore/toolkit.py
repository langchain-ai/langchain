"""Toolkit for interacting with a vector store."""
from typing import List

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


class VectorStoreInfo(BaseModel):
    """Information about a vectorstore."""

    vectorstore: VectorStore = Field(exclude=True)
    name: str
    description: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a vector store."""

    vectorstore_info: VectorStoreInfo = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        description = VectorStoreQATool.get_description(
            self.vectorstore_info.name, self.vectorstore_info.description
        )
        qa_tool = VectorStoreQATool(
            name=self.vectorstore_info.name,
            description=description,
            vectorstore=self.vectorstore_info.vectorstore,
            llm=self.llm,
        )
        description = VectorStoreQAWithSourcesTool.get_description(
            self.vectorstore_info.name, self.vectorstore_info.description
        )
        qa_with_sources_tool = VectorStoreQAWithSourcesTool(
            name=f"{self.vectorstore_info.name}_with_sources",
            description=description,
            vectorstore=self.vectorstore_info.vectorstore,
            llm=self.llm,
        )
        return [qa_tool, qa_with_sources_tool]


class VectorStoreRouterToolkit(BaseToolkit):
    """Toolkit for routing between vectorstores."""

    vectorstores: List[VectorStoreInfo] = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        for vectorstore_info in self.vectorstores:
            description = VectorStoreQATool.get_description(
                vectorstore_info.name, vectorstore_info.description
            )
            qa_tool = VectorStoreQATool(
                name=vectorstore_info.name,
                description=description,
                vectorstore=vectorstore_info.vectorstore,
                llm=self.llm,
            )
            tools.append(qa_tool)
        return tools
