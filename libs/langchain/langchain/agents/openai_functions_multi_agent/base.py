"""Module implements an agent that uses OpenAI's APIs function enabled API."""
import json
from json import JSONDecodeError
from typing import Any, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BasePromptTemplate,
    OutputParserException,
)
from langchain.schema.agent import AgentActionMessageLog
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain.tools import BaseTool

# For backwards compatibility
_FunctionsAgentAction = AgentActionMessageLog


def _parse_ai_message(message: BaseMessage) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    function_call = message.additional_kwargs.get("function_call", {})

    if function_call:
        try:
            arguments = json.loads(function_call["arguments"])
        except JSONDecodeError:
            raise OutputParserException(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` is not valid JSON."
            )

        try:
            tools = arguments["actions"]
        except (TypeError, KeyError):
            raise OutputParserException(
                f"Could not parse tool input: {function_call} because "
                f"the `arguments` JSON does not contain `actions` key."
            )

        final_tools: List[AgentAction] = []
        for tool_schema in tools:
            _tool_input = tool_schema["action"]
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
        return_values={"output": message.content}, log=str(message.content)
    )


class OpenAIMultiFunctionsAgent(BaseMultiActionAgent):
    """An Agent driven by OpenAIs function powered API.

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

    def get_allowed_tools(self) -> List[str]:
        """Get allowed tools."""
        return [t.name for t in self.tools]

    @root_validator
    def validate_llm(cls, values: dict) -> dict:
        if not isinstance(values["llm"], ChatOpenAI):
            raise ValueError("Only supported with ChatOpenAI models.")
        return values

    @root_validator
    def validate_prompt(cls, values: dict) -> dict:
        prompt: BasePromptTemplate = values["prompt"]
        if "agent_scratchpad" not in prompt.input_variables:
            raise ValueError(
                "`agent_scratchpad` should be one of the variables in the prompt, "
                f"got {prompt.input_variables}"
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Get input keys. Input refers to user input here."""
        return ["input"]

    @property
    def functions(self) -> List[dict]:
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
                    }
                },
                "required": ["actions"],
            },
        }
        return [tool_selection]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date, along with observations
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
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
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
            messages, functions=self.functions, callbacks=callbacks
        )
        agent_decision = _parse_ai_message(predicted_message)
        return agent_decision

    @classmethod
    def create_prompt(
        cls,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
    ) -> BasePromptTemplate:
        """Create prompt for this agent.

        Args:
            system_message: Message to use as the system message that will be the
                first in the prompt.
            extra_prompt_messages: Prompt messages that will be placed between the
                system message and the new human input.

        Returns:
            A prompt template to pass into this agent.
        """
        _prompts = extra_prompt_messages or []
        messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]
        if system_message:
            messages = [system_message]
        else:
            messages = []

        messages.extend(
            [
                *_prompts,
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return ChatPromptTemplate(messages=messages)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
        system_message: Optional[SystemMessage] = SystemMessage(
            content="You are a helpful AI assistant."
        ),
        **kwargs: Any,
    ) -> BaseMultiActionAgent:
        """Construct an agent from an LLM and tools."""
        prompt = cls.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message,
        )
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
