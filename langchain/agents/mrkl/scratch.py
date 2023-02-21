from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.agents.mrkl.prompt import SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool, BaseToolkit
from langchain.tools.lookup.toolkit import VectorBackedToolkit

FINAL_ANSWER_ACTION = "Final Answer:"
PREFIX = """Answer the following questions as best you can.
You have access to a variety of tools to ascertain the answer.
If the question is too ambiguous, you may inquire for more information."""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, some examples include [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class ZeroShotToolkitAgent(Agent):
    """Agent for the MRKL chain."""

    lookup_toolkit: BaseToolkit

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "zero-shot-react-description"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

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
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        # tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join([prefix, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        print(template)
        return PromptTemplate(template=template, input_variables=input_variables)

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
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
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
        embeddings = kwargs.get("embeddings", None)
        if embeddings is None:
            raise ValueError("Must provide embeddings to use this agent.")
        lookup_toolkit = VectorBackedToolkit(embeddings=embeddings, tools=tools)
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            lookup_toolkit=lookup_toolkit,
            **kwargs,
        )

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract the tool and input from the text."""
        if FINAL_ANSWER_ACTION in text:
            return "Final Answer", text.split(FINAL_ANSWER_ACTION)[-1].strip()
        # TODO: Maybe add support for just Extract tool and then Generate Tool Input
        tool = self.lookup_toolkit.recommend_tool(text)
        if tool is None:
            print("NO TOOL FOUND", text)
            return None
        regex = r"Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if match is None:
            print("NO MATCH FOUND", text)
            return None
        return [tool.name, match.group(1)]
