"""Module implements an agent that uses OpenAI's APIs function enabled API."""

import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any, Optional, Union

from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.tools import BaseTool
from pydantic import model_validator
from typing_extensions import Self

from langchain.agents import BaseMultiActionAgent
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)

# For backwards compatibility
_FunctionsAgentAction = AgentActionMessageLog


def _parse_ai_message(message: BaseMessage) -> Union[list[AgentAction], AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        msg = f"Expected an AI message got {type(message)}"
        raise TypeError(msg)

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        try:
            arguments = json.loads(function_call["arguments"], strict=False)
        except JSONDecodeError as e:
            msg = (
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )
            raise OutputParserException(msg) from e

        try:
            tools = arguments["actions"]
        except (TypeError, KeyError) as e:
            msg = (
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` JSON does not contain `actions` key."
            )
            raise OutputParserException(msg) from e

        final_tools: list[AgentAction] = []
        for tool_schema in tools:
            if "action" in tool_schema:
                _tool_input = tool_schema["action"]
            else:
                # drop action_name from schema
                _tool_input = tool_schema.copy()
                del _tool_input["action_name"]
            function_name = tool_schema["action_name"]

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

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            _tool = _FunctionsAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )
            final_tools.append(_tool)
        return final_tools

    return AgentFinish(
        return_values={"output": message.content},
        log=str(message.content),
    )


_NOT_SET = object()


@deprecated("0.1.0", alternative="create_openai_tools_agent", removal="1.0")
class OpenAIMultiFunctionsAgent(BaseMultiActionAgent):
    """Agent driven by OpenAIs function powered API.

    Args:
        llm: This should be an instance of ChatOpenAI, specifically a model
            that supports using `functions`.
        tools: The tools this agent has access to.
        prompt: The prompt for this agent, should support agent_scratchpad as one
            of the variables. For an easy way to construct this prompt, use
            `OpenAIMultiFunctionsAgent.create_prompt(...)`
    """

    llm: BaseLanguageModel
    tools: Sequence[BaseTool]
    prompt: BasePromptTemplate

    def get_allowed_tools(self) -> list[str]:
        """Get allowed tools."""
        return [t.name for t in self.tools]

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        prompt: BasePromptTemplate = self.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            msg = (
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
            raise ValueError(msg)
        return self

    @property
    def input_keys(self) -> list[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    @property
    def functions(self) -> list[dict]:
        """Get the functions for the agent."""
        enum_vals = [t.name for t in self.tools]
        tool_selection = {
            # OpenAI functions returns a single tool invocation
            # Here we force the single tool invocation it returns to
            # itself be a list of tool invocations. We do this by constructing
            # a new tool that has one argument which is a list of tools
            # to use.
            "name": "tool_selection",
            "description": "A list of actions to take.",
            "parameters": {
                "title": "tool_selection",
                "description": "A list of actions to take.",
                "type": "object",
                "properties": {
                    "actions": {
                        "title": "actions",
                        "type": "array",
                        "items": {
                            # This is a custom item which bundles the action_name
                            # and the action. We do this because some actions
                            # could have the same schema, and without this there
                            # is no way to differentiate them.
                            "title": "tool_call",
                            "type": "object",
                            "properties": {
                                # This is the name of the action to take
                                "action_name": {
                                    "title": "action_name",
                                    "enum": enum_vals,
                                    "type": "string",
                                    "description": (
                                        "Name of the action to take. The name "
                                        "provided here should match up with the "
                                        "parameters for the action below."
                                    ),
                                },
                                # This is the action to take.
                                "action": {
                                    "title": "Action",
                                    "anyOf": [
                                        {
                                            "title": t.name,
                                            "type": "object",
                                            "properties": t.args,
                                        }
                                        for t in self.tools
                                    ],
                                },
                            },
                            "required": ["action_name", "action"],
                        },
                    },
                },
                "required": ["actions"],
            },
        }
        return [tool_selection]

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[list[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use. Default is None.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = self.llm.predict_messages(
            messages,
            functions=self.functions,
            callbacks=callbacks,
        )
        return _parse_ai_message(predicted_message)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[list[AgentAction], AgentFinish]:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to use. Default is None.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
        selected_inputs = {
            k: kwargs[k] for k in self.prompt.input_variables if k != "agent_scratchpad"
        }
        full_inputs = dict(**selected_inputs, agent_scratchpad=agent_scratchpad)
        prompt = self.prompt.format_prompt(**full_inputs)
        messages = prompt.to_messages()
        predicted_message = await self.llm.apredict_messages(
            messages,
            functions=self.functions,
            callbacks=callbacks,
        )
        return _parse_ai_message(predicted_message)

    @classmethod
    def create_prompt(
        cls,
        system_message: Optional[SystemMessage] = _NOT_SET,  # type: ignore[assignment]
        extra_prompt_messages: Optional[list[BaseMessagePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        """Create prompt for this agent.

        Args:
            system_message: Message to use as the system message that will be the
                first in the prompt.
            extra_prompt_messages: Prompt messages that will be placed between the
                system message and the new human input. Default is None.

        Returns:
            A prompt template to pass into this agent.
        """
        _prompts = extra_prompt_messages or []
        system_message_ = (
            system_message
            if system_message is not _NOT_SET
            else SystemMessage(content="You are a helpful AI assistant.")
        )
        messages: list[Union[BaseMessagePromptTemplate, BaseMessage]]
        messages = [system_message_] if system_message_ else []

        messages.extend(
            [
                *_prompts,
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        return ChatPromptTemplate(messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt_messages: Optional[list[BaseMessagePromptTemplate]] = None,
        system_message: Optional[SystemMessage] = _NOT_SET,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> BaseMultiActionAgent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: The language model to use.
            tools: A list of tools to use.
            callback_manager: The callback manager to use. Default is None.
            extra_prompt_messages: Extra prompt messages to use. Default is None.
            system_message: The system message to use.
                Default is a default system message.
            kwargs: Additional arguments.
        """
        system_message_ = (
            system_message
            if system_message is not _NOT_SET
            else SystemMessage(content="You are a helpful AI assistant.")
        )
        prompt = cls.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message_,
        )
        return cls(  # type: ignore[call-arg]
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
