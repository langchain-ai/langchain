from __future__ import annotations

from typing import List, Union

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage


class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""

    def __init__(self, tool: str, tool_input: Union[str, dict], log: str, **kwargs):
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @property
    def lc_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return True

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""


class AgentActionMessageLog(AgentAction):
    message_log: List[BaseMessage]


class AgentFinish(Serializable):
    """The final return value of an ActionAgent."""

    def __init__(self, return_values: dict, log: str, **kwargs):
        super().__init__(return_values=return_values, log=log, **kwargs)

    @property
    def lc_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return True

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value"""
