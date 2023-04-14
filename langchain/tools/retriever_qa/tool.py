"""Tools for interacting with retriever."""
import json
from typing import Any, Dict

from pydantic import BaseModel, Field

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA, BaseRetrievalQA
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.tools.base import BaseTool
from langchain.schema import BaseRetriever

class BaseRetrievalQAInfo(BaseModel):
    """Base class to store information"""
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))
    retriever: BaseRetriever = Field(exclude=True)

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
                "Output is a json serialized dictionary with keys `answer` and `sources`. "
                "Only use this tool if the user explicitly asks for sources. "
            )
            template: str = base_template + sources_template
        else:
            template:  str = base_template
        return template
    
    
class RetrievalQATool(BaseRetrievalQAInfo, BaseTool):
    """Tool for the RetrievalQA Chain. To be initialized with name and chain."""
    
    @staticmethod
    def get_description(name: str, decription: str) -> str:
        template: str = self._get_template(
            name=name,
            description=description,
            include_sources=False
        )
        return template
    
    def _load_chain(self, **kwargs) -> BaseRetrievalQA:
        '''load the chain'''
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            **kwargs
        )
        return chain

    def _run(self, query: str, **kwargs: Any) -> str:
        """Use the tool."""
        chain = self._load_chain(**kwargs)
        result = chain.run(query)
        return result
    
    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Use the tool asynchronously."""
        chain = self._load_chain(**kwargs)
        result = await chain.arun(query)
        return result
    
class RetrievalQAWithSourcesTool(BaseRetrievalQAInfo, BaseTool):
    """Tool for the RetrievalQAWithSources chain."""
    
    @staticmethod
    def get_description(name: str, decription: str) -> str:
        template: str = self._get_template(
            name=name,
            description=description,
            include_sources=True
        )
        return template

    def _load_chain(self, **kwargs) -> BaseQAWithSourcesChain:
        '''load the chain'''
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            **kwargs
        )
        return chain

    def _run(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Use the tool."""
        chain = self._load_chain(**kwargs)
        result = chain(
            inputs={chain.question_key: query},
            return_only_outputs=True
        )
        return json.dumps(result)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Use the tool asynchronously"""
        chain = self._load_chain(**kwargs)
        result = await chain(
            inputs={chain.question_key: query},
            return_only_outputs=True
        )
        return json.dumps(result)