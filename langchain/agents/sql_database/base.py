"""Agent for interacting with a SQL database."""
import string
from typing import Any, List, Optional, Sequence

from langchain import PromptTemplate
from langchain.agents import ZeroShotAgent
from langchain.agents.mrkl.base import create_zero_shot_prompt
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.sql_database.prompt import PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool
from langchain.tools.sql_database.toolkit import SQLDatabaseToolkit


class SQLDatabaseAgent(ZeroShotAgent):
    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        dialect: str = "sqlite",
        top_k: int = 10,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
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
            format_instructions, input_variables, prefix, suffix, tools
        )

    @classmethod
    def from_llm_and_toolkit(
        cls,
        llm: BaseLLM,
        toolkit: SQLDatabaseToolkit,
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        top_k: int = 10,
        **kwargs: Any,
    ):
        """Create an agent from an LLM, database, and tools."""
        tools = toolkit.get_tools()
        prompt = cls.create_prompt(
            tools,
            dialect=toolkit.dialect,
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
