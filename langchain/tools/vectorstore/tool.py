"""Tools for interacting with vectorstores."""

import json

from pydantic import BaseModel, Field

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStore


class BaseVectorStoreTool(BaseModel):
    """Base class for tools that use a VectorStore."""

    vectorstore: VectorStore = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @staticmethod
    def _get_template(name: str, description: str, include_sources: bool) -> str:
        base_template: str = (
            f"Userful for when you need to answer questions about {name}. "
            f"Whenever you need information about {description} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )

        if include_sources:
            sources_template: str = (
                "Output is a json serialized dictionary "
                "with keys `answer` and `sources`. "
                "Only use this tool if the user explicitly asks for sources. "
            )
            template = base_template + sources_template
        else:
            template = base_template
        return template


class VectorStoreQATool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQA chain. To be initialized with name and chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """get description from template"""
        template: str = BaseVectorStoreTool._get_template(
            name=name, description=description, include_sources=False
        )
        return template

    def _load_chain(self) -> BaseRetrievalQA:
        """load the chain"""
        chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.vectorstore.as_retriever()
        )
        return chain

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = self._load_chain()
        result = chain.run(query)
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        chain = self._load_chain()
        result = await chain.arun(query)
        return result


class VectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQAWithSources chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """get description from template"""
        template: str = (
            "Useful for when you need to answer questions about {name} and the sources "
            "used to construct the answer. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            " Input should be a fully formed question. "
            "Output is a json serialized dictionary with keys `answer` and `sources`. "
            "Only use this tool if the user explicitly asks for sources."
        )
        return template.format(name=name, description=description)

    def _load_chain(self) -> BaseQAWithSourcesChain:
        """load the chain"""
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm, retriever=self.vectorstore.as_retriever()
        )
        return chain

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = self._load_chain()
        result = chain(inputs={chain.question_key: query}, return_only_outputs=True)
        return json.dumps(result)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        chain = self._load_chain()
        result = await chain.arun(
            inputs={chain.question_key: query}, return_only_outputs=True
        )
        return json.dumps(result)
