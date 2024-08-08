"""Toolkit for interacting with a vector store."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.vectorstores import VectorStore


class VectorStoreInfo(BaseModel):
    """Information about a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    name: str
    description: str

    class Config:
        arbitrary_types_allowed = True


class VectorStoreToolkit(BaseToolkit):
    """Toolkit for interacting with a Vector Store."""

    vectorstore_info: VectorStoreInfo = Field(exclude=True)
    llm: BaseLanguageModel

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        try:
            from langchain_community.tools.vectorstore.tool import (
                VectorStoreQATool,
                VectorStoreQAWithSourcesTool,
            )
        except ImportError:
            raise ImportError(
                "You need to install langchain-community to use this toolkit."
            )
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
    """Toolkit for routing between Vector Stores."""

    vectorstores: List[VectorStoreInfo] = Field(exclude=True)
    llm: BaseLanguageModel

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        try:
            from langchain_community.tools.vectorstore.tool import (
                VectorStoreQATool,
            )
        except ImportError:
            raise ImportError(
                "You need to install langchain-community to use this toolkit."
            )
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
