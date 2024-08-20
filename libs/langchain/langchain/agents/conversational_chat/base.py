"""An agent designed to hold a conversation in addition to using tools."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.agents.conversational_chat.prompt import (
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)
from langchain.agents.utils import validate_tools_single_input
from langchain.chains import LLMChain


@deprecated("0.1.0", alternative="create_json_chat_agent", removal="1.0")
class ConversationalChatAgent(Agent):
    """An agent designed to hold a conversation in addition to using tools."""

    output_parser: AgentOutputParser = Field(default_factory=ConvoOutputParser)
    """Output parser for the agent."""
    template_tool_response: str = TEMPLATE_TOOL_RESPONSE
    """Template for the tool response."""

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ConvoOutputParser()

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with.

        Returns:
            "Observation: "
        """
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with.

        Returns:
            "Thought: "
        """
        return "Thought:"

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        super()._validate_tools(tools)
        validate_tools_single_input(cls.__name__, tools)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        """Create a prompt for the agent.

        Args:
            tools: The tools to use.
            system_message: The system message to use.
                Defaults to the PREFIX.
            human_message: The human message to use.
                Defaults to the SUFFIX.
            input_variables: The input variables to use. Defaults to None.
            output_parser: The output parser to use. Defaults to None.

        Returns:
            A PromptTemplate.
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions()
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)  # type: ignore[arg-type]

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=self.template_tool_response.format(observation=observation)
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The language model to use.
            tools: A list of tools to use.
            callback_manager: The callback manager to use. Default is None.
            output_parser: The output parser to use. Default is None.
            system_message: The system message to use. Default is PREFIX.
            human_message: The human message to use. Default is SUFFIX.
            input_variables: The input variables to use. Default is None.
            **kwargs: Any additional arguments.

        Returns:
            An agent.
        """
        cls._validate_tools(tools)
        _output_parser = output_parser or cls._get_default_output_parser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        llm_chain = LLMChain(  # type: ignore[misc]
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
