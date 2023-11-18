from __future__ import annotations

from typing import Any, Literal, Sequence, Union

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage


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
