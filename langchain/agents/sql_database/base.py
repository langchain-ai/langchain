"""Agent for interacting with a SQL database."""
import string
from typing import Any, List, Optional, Sequence

from langchain.agents.agent import Agent
from langchain.agents.mrkl.base import ZeroShotAgent, create_zero_shot_prompt
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.sql_database.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool


class SQLDatabaseAgent(ZeroShotAgent):
    """Agent for interacting with a SQL database."""

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        dialect: str = "sqlite",
        top_k: int = 10,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            dialect: The dialect of the SQL database.
            top_k: The number of results to return.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            format_instructions: Instructions to put before the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        prefix = string.Formatter().format(prefix, dialect=dialect, top_k=top_k)
        return create_zero_shot_prompt(
            tools, prefix, suffix, format_instructions, input_variables
        )

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        dialect: str = "sqlite",
        top_k: int = 10,
        **kwargs: Any,
    ) -> Agent:
        """Create an agent from an LLM, database, and tools."""
        prompt = cls.create_prompt(
            tools,
            dialect=dialect,
            top_k=top_k,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
