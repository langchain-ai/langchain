"""Toolkit for interacting with a Power BI dataset."""
from typing import List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.chains.llm import LLMChain
from langchain.powerbi import PowerBIDataset
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.tools.powerbi.prompt import QUERY_CHECKER
from langchain.tools.powerbi.tool import (
    InfoPowerBITool,
    ListPowerBITool,
    QueryCheckerTool,
    QueryPowerBITool,
)


class PowerBIToolkit(BaseToolkit):
    """Toolkit for interacting with PowerBI dataset."""

    powerbi: PowerBIDataset = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        if self.llm is None:
            llmc = None
        else:
            llmc = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["tool_input"]
                ),
            )
        return [
            QueryPowerBITool(powerbi=self.powerbi),
            InfoPowerBITool(powerbi=self.powerbi),
            ListPowerBITool(powerbi=self.powerbi),
            QueryCheckerTool(powerbi=self.powerbi, llm_chain=llmc),
        ]
