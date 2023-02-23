"""Agent for interacting with JSON."""
import string
from typing import Any, List, Optional, Sequence

from langchain.agents.agent import Agent
from langchain.agents.json.prompt import PREFIX, SUFFIX
from langchain.agents.mrkl.base import ZeroShotAgent, create_zero_shot_prompt
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool


class JsonAgent(ZeroShotAgent):
    """Agent for interacting with JSON."""

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            format_instructions: Instructions to put before the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
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
        **kwargs: Any,
    ) -> Agent:
        """Create an agent from an LLM and a list of tools."""
        prompt = cls.create_prompt(
            tools, prefix, suffix, format_instructions, input_variables
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
