"""Tools for interacting with retriever."""
import json

from pydantic import Field

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain.llms.base import BaseLLM
from langchain.schema import BaseRetriever
from langchain.tools.base import BaseTool


class BaseRetrievalQAInfo(BaseTool):
    """Base class to store information"""

    llm: BaseLLM = Field(exclude=True)
    retriever: BaseRetriever = Field(exclude=True)

    class Config(BaseTool):
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


class RetrievalQATool(BaseRetrievalQAInfo, BaseTool):
    """Tool for the RetrievalQA Chain. To be initialized with name and chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = BaseRetrievalQAInfo._get_template(
            name=name, description=description, include_sources=False
        )
        return template

    def _load_chain(self) -> BaseRetrievalQA:
        """load the chain"""
        chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
        return chain

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = self._load_chain()
        result = chain.run(query)
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Chain, RetrievalQATool does not support async")


class RetrievalQAWithSourcesTool(BaseRetrievalQAInfo, BaseTool):
    """Tool for the RetrievalQAWithSources chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = BaseRetrievalQAInfo._get_template(
            name=name, description=description, include_sources=True
        )
        return template

    def _load_chain(self) -> BaseQAWithSourcesChain:
        """load the chain"""
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm, retriever=self.retriever
        )
        return chain

    def _run(self, query: str) -> str:
        """Use the tool."""
        chain = self._load_chain()
        result = chain(inputs={chain.question_key: query}, return_only_outputs=True)
        return json.dumps(result)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously"""
        raise NotImplementedError(
            "Chain, RetrievalQAWithSourcesTool does not support async"
        )
