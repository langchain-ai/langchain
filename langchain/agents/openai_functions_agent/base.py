"""Module implements an agent that uses OpenAI's APIs function enabled API."""
import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseSingleActionAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    Callbacks,
)
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain.tools import BaseTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function


@dataclass
class _FunctionsAgentAction(AgentAction):
    message_log: List[BaseMessage]


def _convert_agent_action_to_messages(agent_action: AgentAction) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if not isinstance(agent_action, _FunctionsAgentAction):
        raise ValueError("This agent type only works with _FunctionsAgentAction")
    return agent_action.message_log


def _create_function_message(
    agent_action: AgentAction, observation: str
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        content = json.dumps(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )


def _format_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Format intermediate steps.
    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations
    Returns:
        list of messages to send to the LLM for the next prediction
    """
    messages = []

    for intermediate_step in intermediate_steps:
        agent_action, observation = intermediate_step
        messages.extend(_convert_agent_action_to_messages(agent_action))
        messages.append(_create_function_message(agent_action, observation))

    return messages


def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        function_call = message.additional_kwargs["function_call"]
        function_name = function_call["name"]
        try:
            _tool_input = json.loads(function_call["arguments"])
        except JSONDecodeError:
            raise ValueError(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = "responded: {content}\n" if message.content else "\n"

        return _FunctionsAgentAction(
            tool=function_name,
            tool_input=tool_input,
            log=f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n",
            message_log=[message],
        )

    return AgentFinish(return_values={"output": message.content}, log=message.content)


class OpenAIFunctionsAgent(BaseSingleActionAgent):
    """An Agent driven by OpenAIs function powered API."""

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate

    def get_allowed_tools(self) -> List[str]:
        """Get allowed tools."""
        return list([t.name for t in self.tools])

    @property
    def input_keys(self) -> List[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    @property
    def functions(self) -> List[dict]:
        return [dict(format_tool_to_openai_function(t)) for t in self.tools]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
            **kwargs: User inputs.
        Returns:
            Action specifying what tool to use.
        """
        user_input = kwargs["input"]
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        prompt = self.prompt.format_prompt(
            input=user_input, agent_scratchpad=agent_scratchpad
        )
        messages = prompt.to_messages()
        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.
        Returns:
            Action specifying what tool to use.
        """
        user_input = kwargs["input"]
        agent_scratchpad = _format_intermediate_steps(intermediate_steps)
        prompt = self.prompt.format_prompt(
            input=user_input, agent_scratchpad=agent_scratchpad
        )
        messages = prompt.to_messages()
        predicted_message = await self.llm.apredict_messages(
            messages, functions=self.functions
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision

    @classmethod
    def create_prompt(cls) -> BasePromptTemplate:
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        input_variables = ["input", "agent_scratchpad"]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools."""
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("Only supported with OpenAI models.")
        prompt = cls.create_prompt()
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
