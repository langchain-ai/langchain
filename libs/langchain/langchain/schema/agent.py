from __future__ import annotations

from typing import NamedTuple, Union

from langchain.load.serializable import Serializable


class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is LangChain serializable."""
        return True


class AgentFinish(NamedTuple):
    """The final return value of an ActionAgent."""

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value"""
