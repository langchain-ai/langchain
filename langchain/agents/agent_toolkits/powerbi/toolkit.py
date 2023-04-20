"""Toolkit for interacting with a Power BI dataset."""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.tools.powerbi.prompt import QUESTION_TO_QUERY
from langchain.tools.powerbi.tool import (
    InfoPowerBITool,
    InputToQueryTool,
    ListPowerBITool,
    QueryPowerBITool,
)
from langchain.utilities.powerbi import PowerBIDataset


class PowerBIToolkit(BaseToolkit):
    """Toolkit for interacting with PowerBI dataset."""

    powerbi: PowerBIDataset = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)
    examples: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        chain = (
            LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    template=QUESTION_TO_QUERY,
                    input_variables=["tool_input", "tables", "schemas", "examples"],
                ),
            ),
        )
        return [
            QueryPowerBITool(powerbi=self.powerbi),
            InfoPowerBITool(powerbi=self.powerbi),
            ListPowerBITool(powerbi=self.powerbi),
            InputToQueryTool(
                powerbi=self.powerbi, llm_chain=chain, examples=self.examples
            ),
        ]
