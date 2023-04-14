"""Toolkit for interacting with a vector store."""
from typing import List, Optional, Tuple, Type, Union, cast

from pydantic import BaseModel, Field, validator

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.schema import BaseRetriever
from langchain.tools import BaseTool
from langchain.tools.retrieval_qa.tool import (
    BaseRetrievalQAInfo,
    RetrievalQATool,
    RetrievalQAWithSourcesTool,
)


class RetrievalQAToolkit(BaseToolkit):
    """Wrapper Toolkit for RetrievalQATool, RetrievalQAWithSourcesTool"""

    retrieval_qa_info: BaseRetrievalQAInfo = Field(exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> list[BaseTool]:
        """Use static method from BaseRetrievalQAInfo"""
        name, description, retriever = cast(
            Tuple[str, str, BaseRetriever],
            (
                self.retrieval_qa_info.name,
                self.retrieval_qa_info.description,
                self.retrieval_qa_info.retriever,
            ),
        )

        qa_tool = RetrievalQATool(
            name=name,
            description=RetrievalQATool.get_description(
                name=name, description=description
            ),
            retriever=retriever,
            llm=self.llm,
        )

        qa_with_sources_tool = RetrievalQAWithSourcesTool(
            name=f"{name}_with_sources",
            description=RetrievalQAWithSourcesTool.get_description(
                name=name, description=description
            ),
            retriever=retriever,
            llm=self.llm,
        )

        return [qa_tool, qa_with_sources_tool]


class RetrievalQARouterToolkit(BaseToolkit):
    """Toolkit for routing among Retriever QA related tools."""

    retrieval_qa_info: List[BaseRetrievalQAInfo] = Field(..., min_items=1, exclude=True)
    llm: BaseLLM = Field(default_factory=lambda: OpenAI(temperature=0))
    include_sources: Optional[bool] = False

    @validator("retrieval_qa_info", pre=True)
    def _convert_to_list(
        cls, value: Union[BaseRetrievalQAInfo, List[BaseRetrievalQAInfo]]
    ) -> Union[BaseRetrievalQAInfo, List[BaseRetrievalQAInfo]]:
        """Convert BaseRetrievalQAInfo to List[BaseRetrievalQAInfo]"""
        if isinstance(value, BaseRetrievalQAInfo):
            return [value]
        return value

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _make_tool(self, rq_info: BaseRetrievalQAInfo) -> BaseTool:
        base_retrieval_qa_tool = (
            RetrievalQAWithSourcesTool if self.include_sources else RetrievalQATool
        )
        qa_tool = base_retrieval_qa_tool(
            name=rq_info.name,
            description=rq_info.description,
            retriever=rq_info.retriever,
            llm=self.llm,
        )
        return qa_tool

    def get_tools(self) -> List[BaseTool]:
        return [self._make_tool(rq_info=rq) for rq in self.retrieval_qa_info]
