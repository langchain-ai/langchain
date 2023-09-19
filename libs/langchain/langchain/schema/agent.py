from __future__ import annotations

from typing import Union

from langchain.load.serializable import Serializable


class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""

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


class AgentFinish(Serializable):
    """The final return value of an ActionAgent."""

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
