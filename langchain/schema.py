"""Common schema objects."""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

AGENT_FINISH_OBSERVATION = "__agent_finish__"


@dataclass
class AgentAction:
    """Agent's action to take."""

    tool: str
    tool_input: str
    log: str


@dataclass
class AgentFinish(AgentAction):
    """Agent's return value."""

    return_values: dict


class Generation(NamedTuple):
    """Output of a single generation."""

    text: str
    """Generated text output."""
    # TODO: add log probs


class LLMResult(NamedTuple):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""
