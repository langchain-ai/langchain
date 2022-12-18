"""Common schema objects."""

from typing import NamedTuple


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: str
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class Generation(NamedTuple):
    """Output of a single generation."""

    text: str
    """Generated text output."""
    # TODO: add log probs
