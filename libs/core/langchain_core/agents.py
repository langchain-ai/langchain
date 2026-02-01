"""Schema definitions for representing agent actions, observations, and return values.

!!! warning

    The schema definitions are provided for backwards compatibility.

!!! warning

    New agents should be built using the
    [`langchain` library](https://pypi.org/project/langchain/), which provides a
    simpler and more flexible way to define agents.

    See docs on [building agents](https://docs.langchain.com/oss/python/langchain/agents).

Agents use language models to choose a sequence of actions to take.

A basic agent works in the following manner:

1. Given a prompt an agent uses an LLM to request an action to take
    (e.g., a tool to run).
2. The agent executes the action (e.g., runs the tool), and receives an observation.
3. The agent returns the observation to the LLM, which can then be used to generate
    the next action.
4. When the agent reaches a stopping condition, it returns a final return value.

The schemas for the agents themselves are defined in `langchain.agents.agent`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)


class AgentAction(Serializable):
    """Represents a request to execute an action by an agent.

    The action consists of the name of the tool to execute and the input to pass
    to the tool. The log is used to pass along extra information about the action.
    """

    tool: str
    """The name of the `Tool` to execute."""

    tool_input: str | dict
    """The input to pass in to the `Tool`."""

    log: str
    """Additional information to log about the action.

    This log can be used in a few ways. First, it can be used to audit what exactly the
    LLM predicted to lead to this `(tool, tool_input)`.

    Second, it can be used in future iterations to show the LLMs prior thoughts. This is
    useful when `(tool, tool_input)` does not contain full information about the LLM
    prediction (for example, any `thought` before the tool/tool_input).
    """

    type: Literal["AgentAction"] = "AgentAction"

    # Override init to support instantiation by position for backward compat.
    def __init__(self, tool: str, tool_input: str | dict, log: str, **kwargs: Any):
        """Create an `AgentAction`.

        Args:
            tool: The name of the tool to execute.
            tool_input: The input to pass in to the `Tool`.
            log: Additional information to log about the action.
        """
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`AgentAction` is serializable.

        Returns:
            `True`
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "agent"]`
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Return the messages that correspond to this action."""
        return _convert_agent_action_to_messages(self)


class AgentActionMessageLog(AgentAction):
    """Representation of an action to be executed by an agent.

    This is similar to `AgentAction`, but includes a message log consisting of
    chat messages.

    This is useful when working with `ChatModels`, and is used to reconstruct
    conversation history from the agent's perspective.
    """

    message_log: Sequence[BaseMessage]
    """Similar to log, this can be used to pass along extra information about what exact
    messages were predicted by the LLM before parsing out the `(tool, tool_input)`.

    This is again useful if `(tool, tool_input)` cannot be used to fully recreate the
    LLM prediction, and you need that LLM prediction (for future agent iteration).

    Compared to `log`, this is useful when the underlying LLM is a chat model (and
    therefore returns messages rather than a string).
    """
    # Ignoring type because we're overriding the type from AgentAction.
    # And this is the correct thing to do in this case.
    # The type literal is used for serialization purposes.
    type: Literal["AgentActionMessageLog"] = "AgentActionMessageLog"  # type: ignore[assignment]


class AgentStep(Serializable):
    """Result of running an `AgentAction`."""

    action: AgentAction
    """The `AgentAction` that was executed."""

    observation: Any
    """The result of the `AgentAction`."""

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Messages that correspond to this observation."""
        return _convert_agent_observation_to_messages(self.action, self.observation)


class AgentFinish(Serializable):
    """Final return value of an `ActionAgent`.

    Agents return an `AgentFinish` when they have reached a stopping condition.
    """

    return_values: dict
    """Dictionary of return values."""

    log: str
    """Additional information to log about the return value.

    This is used to pass along the full LLM prediction, not just the parsed out
    return value.

    For example, if the full LLM prediction was `Final Answer: 2` you may want to just
    return `2` as a return value, but pass along the full string as a `log` (for
    debugging or observability purposes).
    """
    type: Literal["AgentFinish"] = "AgentFinish"

    def __init__(self, return_values: dict, log: str, **kwargs: Any):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(return_values=return_values, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "agent"]`
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """Messages that correspond to this observation."""
        return [AIMessage(content=self.log)]


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
) -> Sequence[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        `AIMessage` that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return agent_action.message_log
    return [AIMessage(content=agent_action.log)]


def _convert_agent_observation_to_messages(
    agent_action: AgentAction, observation: Any
) -> Sequence[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.
        observation: Observation to convert to a message.

    Returns:
        `AIMessage` that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [_create_function_message(agent_action, observation)]
    content = observation
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    return [HumanMessage(content=content)]


def _create_function_message(
    agent_action: AgentAction, observation: Any
) -> FunctionMessage:
    """Convert agent action and observation into a function message.

    Args:
        agent_action: the tool invocation request from the agent.
        observation: the result of the tool invocation.

    Returns:
        `FunctionMessage` that corresponds to the original tool invocation.
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
