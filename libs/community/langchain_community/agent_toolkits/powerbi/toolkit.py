"""Toolkit for interacting with a Power BI dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_community.tools.powerbi.prompt import (
    QUESTION_TO_QUERY_BASE,
    SINGLE_QUESTION_TO_QUERY,
    USER_INPUT,
)
from langchain_community.tools.powerbi.tool import (
    InfoPowerBITool,
    ListPowerBITool,
    QueryPowerBITool,
)
from langchain_community.utilities.powerbi import PowerBIDataset

if TYPE_CHECKING:
    from langchain.chains.llm import LLMChain


class PowerBIToolkit(BaseToolkit):
    """Toolkit for interacting with Power BI dataset.

    *Security Note*: This toolkit interacts with an external service.

        Control access to who can use this toolkit.

        Make sure that the capabilities given by this toolkit to the calling
        code are appropriately scoped to the application.

        See https://python.langchain.com/docs/security for more information.

    Parameters:
        powerbi: The Power BI dataset.
        llm: The language model to use.
        examples: Optional. The examples for the prompt. Default is None.
        max_iterations: Optional. The maximum iterations to run. Default is 5.
        callback_manager: Optional. The callback manager. Default is None.
        output_token_limit: The output token limit. Default is 4000.
        tiktoken_model_name: Optional. The TikToken model name. Default is None.
    """

    powerbi: PowerBIDataset = Field(exclude=True)
    llm: Union[BaseLanguageModel, BaseChatModel] = Field(exclude=True)
    examples: Optional[str] = None
    max_iterations: int = 5
    callback_manager: Optional[BaseCallbackManager] = None
    output_token_limit: int = 4000
    tiktoken_model_name: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

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
        from langchain.chains.llm import LLMChain

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
