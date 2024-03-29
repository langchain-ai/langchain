from typing import Any, List, Optional, Sequence, Tuple

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.chat.prompt import (
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
)
from langchain.agents.utils import validate_tools_single_input
from langchain.chains.llm import LLMChain


@deprecated("0.1.0", alternative="create_react_agent", removal="0.2.0")
class ChatAgent(Agent):
    """Chat Agent."""

    output_parser: AgentOutputParser = Field(default_factory=ChatOutputParser)
    """Output parser for the agent."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
        if not isinstance(agent_scratchpad, str):
            raise ValueError("agent_scratchpad should be of type string.")
        if agent_scratchpad:
            return (
                f"This was your previous work "
                f"(but I haven't seen any of it! I only see what "
                f"you return as final answer):\n{agent_scratchpad}"
            )
        else:
            return agent_scratchpad

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ChatOutputParser()

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(class_name=cls.__name__, tools=tools)

    @property
    def _stop(self) -> List[str]:
        return ["Observation:"]

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message_prefix: str = SYSTEM_MESSAGE_PREFIX,
        system_message_suffix: str = SYSTEM_MESSAGE_SUFFIX,
        human_message: str = HUMAN_MESSAGE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = "\n\n".join(
            [
                system_message_prefix,
                tool_strings,
                format_instructions,
                system_message_suffix,
            ]
        )
        messages = [
            SystemMessagePromptTemplate.from_template(template),
            HumanMessagePromptTemplate.from_template(human_message),
        ]
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)  # type: ignore[arg-type]

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        system_message_prefix: str = SYSTEM_MESSAGE_PREFIX,
        system_message_suffix: str = SYSTEM_MESSAGE_SUFFIX,
        human_message: str = HUMAN_MESSAGE,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            system_message_prefix=system_message_prefix,
            system_message_suffix=system_message_suffix,
            human_message=human_message,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError
