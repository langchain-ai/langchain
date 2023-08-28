"""Toolkit for interacting with a Power BI dataset."""
from typing import List, Optional, Union

from langchain_xfyun.agents.agent_toolkits.base import BaseToolkit
from langchain_xfyun.callbacks.base import BaseCallbackManager
from langchain_xfyun.chains.llm import LLMChain
from langchain_xfyun.chat_models.base import BaseChatModel
from langchain_xfyun.prompts import PromptTemplate
from langchain_xfyun.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_xfyun.pydantic_v1 import Field
from langchain_xfyun.schema.language_model import BaseLanguageModel
from langchain_xfyun.tools import BaseTool
from langchain_xfyun.tools.powerbi.prompt import (
    QUESTION_TO_QUERY_BASE,
    SINGLE_QUESTION_TO_QUERY,
    USER_INPUT,
)
from langchain_xfyun.tools.powerbi.tool import (
    InfoPowerBITool,
    ListPowerBITool,
    QueryPowerBITool,
)
from langchain_xfyun.utilities.powerbi import PowerBIDataset


class PowerBIToolkit(BaseToolkit):
    """Toolkit for interacting with Power BI dataset."""

    powerbi: PowerBIDataset = Field(exclude=True)
    llm: Union[BaseLanguageModel, BaseChatModel] = Field(exclude=True)
    examples: Optional[str] = None
    max_iterations: int = 5
    callback_manager: Optional[BaseCallbackManager] = None
    output_token_limit: Optional[int] = None
    tiktoken_model_name: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QueryPowerBITool(
                llm_chain=self._get_chain(),
                powerbi=self.powerbi,
                examples=self.examples,
                max_iterations=self.max_iterations,
                output_token_limit=self.output_token_limit,
                tiktoken_model_name=self.tiktoken_model_name,
            ),
            InfoPowerBITool(powerbi=self.powerbi),
            ListPowerBITool(powerbi=self.powerbi),
        ]

    def _get_chain(self) -> LLMChain:
        """Construct the chain based on the callback manager and model type."""
        if isinstance(self.llm, BaseLanguageModel):
            return LLMChain(
                llm=self.llm,
                callback_manager=self.callback_manager
                if self.callback_manager
                else None,
                prompt=PromptTemplate(
                    template=SINGLE_QUESTION_TO_QUERY,
                    input_variables=["tool_input", "tables", "schemas", "examples"],
                ),
            )

        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=QUESTION_TO_QUERY_BASE,
                input_variables=["tables", "schemas", "examples"],
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=USER_INPUT,
                input_variables=["tool_input"],
            )
        )
        return LLMChain(
            llm=self.llm,
            callback_manager=self.callback_manager if self.callback_manager else None,
            prompt=ChatPromptTemplate.from_messages([system_prompt, human_prompt]),
        )
