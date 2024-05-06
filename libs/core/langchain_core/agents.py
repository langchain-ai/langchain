"""
**Agent** is a class that uses an LLM to choose a sequence of actions to take.

In Chains, a sequence of actions is hardcoded. In Agents,
a language model is used as a reasoning engine to determine which actions
to take and in which order.

Agents select and use **Tools** and **Toolkits** for actions.

**Class hierarchy:**

.. code-block::

    BaseSingleActionAgent --> LLMSingleActionAgent
                              OpenAIFunctionsAgent
                              XMLAgent
                              Agent --> <name>Agent  # Examples: ZeroShotAgent, ChatAgent


    BaseMultiActionAgent  --> OpenAIMultiFunctionsAgent


**Main helpers:**

.. code-block::

    AgentType, AgentExecutor, AgentOutputParser, AgentExecutorIterator,
    AgentAction, AgentFinish, AgentStep

"""  # noqa: E501
from __future__ import annotations

import json
from typing import Any, List, Literal, Sequence, Union

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)


class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action.
    This log can be used in a few ways. First, it can be used to audit
    what exactly the LLM predicted to lead to this (tool, tool_input).
    Second, it can be used in future iterations to show the LLMs prior
    thoughts. This is useful when (tool, tool_input) does not contain
    full information about the LLM prediction (for example, any `thought`
    before the tool/tool_input)."""
    type: Literal["AgentAction"] = "AgentAction"

    def __init__(
        self, tool: str, tool_input: Union[str, dict], log: str, **kwargs: Any
    ):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this action."""
        return _convert_agent_action_to_messages(self)


class AgentActionMessageLog(AgentAction):
    message_log: Sequence[BaseMessage]
    """Similar to log, this can be used to pass along extra
    information about what exact messages were predicted by the LLM
    before parsing out the (tool, tool_input). This is again useful
    if (tool, tool_input) cannot be used to fully recreate the LLM
    prediction, and you need that LLM prediction (for future agent iteration).
    Compared to `log`, this is useful when the underlying LLM is a
    ChatModel (and therefore returns messages rather than a string)."""
    # Ignoring type because we're overriding the type from AgentAction.
    # And this is the correct thing to do in this case.
    # The type literal is used for serialization purposes.
    type: Literal["AgentActionMessageLog"] = "AgentActionMessageLog"  # type: ignore


class AgentStep(Serializable):
    """The result of running an AgentAction."""

    action: AgentAction
    """The AgentAction that was executed."""
    observation: Any
    """The result of the AgentAction."""

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this observation."""
        return _convert_agent_observation_to_messages(self.action, self.observation)


class AgentFinish(Serializable):
    """The final return value of an ActionAgent."""

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value.
    This is used to pass along the full LLM prediction, not just the parsed out
    return value. For example, if the full LLM prediction was
    `Final Answer: 2` you may want to just return `2` as a return value, but pass
    along the full string as a `log` (for debugging or observability purposes).
    """
    type: Literal["AgentFinish"] = "AgentFinish"

    def __init__(self, return_values: dict, log: str, **kwargs: Any):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(return_values=return_values, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this observation."""
        return [AIMessage(content=self.log)]


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
) -> Sequence[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return agent_action.message_log
    else:
        return [AIMessage(content=agent_action.log)]


def _convert_agent_observation_to_messages(
    agent_action: AgentAction, observation: Any
) -> Sequence[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [_create_function_message(agent_action, observation)]
    else:
        return [HumanMessage(content=observation)]


def _create_function_message(
    agent_action: AgentAction, observation: Any
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )
