"""Tool for the VectorDBQA chain."""

from pydantic import BaseModel, root_validator, Field
from langchain.vectorstores.base import VectorStore
from langchain.chains.vector_db_qa.base import VectorDBQA
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain.tools.base import BaseTool


class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class VectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    template: str = (
        "Useful for when you need to answer questions about {name}. "
        "Input should be a fully formed question."
    )
    chain: VectorDBQA
    description = ""

    @root_validator()
    def create_description_from_template(cls, values: dict) -> dict:
        """Create description from template."""
        if "name" in values:
            values["description"] = values["template"].format(name=values["name"])
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")


class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQAWithSources chain. To be initialized with name and chain."""

    template: str = (
        "Useful for when you need to answer questions about {name} and the sources "
        "used to construct the answer. Input should be a fully formed question."
    )
    chain: VectorDBQAWithSourcesChain
    description = ""

    @root_validator()
    def create_description_from_template(cls, values: dict) -> dict:
        """Create description from template."""
        if "name" in values:
            values["description"] = values["template"].format(name=values["name"])
        return values

    def _run(self, query: str) -> str:
        """Use the tool."""
        return self.chain.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("VectorDBQATool does not support async")
