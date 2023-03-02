"""Common schema objects."""

from typing import Any, Dict, List, NamedTuple, Optional

from pydantic import BaseModel


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: str
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class Generation(BaseModel):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""
    """May include things like reason for finishing (e.g. in OpenAI)"""
    # TODO: add log probs


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class ChatMessage(BaseModel):
    """Message object."""

    text: str
    """Generated text output."""

    role: str
    """Role of the chatter."""


class ChatGeneration(BaseModel):
    """Output of a single generation."""

    message: ChatMessage

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""
    """May include things like reason for finishing (e.g. in OpenAI)"""
    # TODO: add log probs


class ChatResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChatGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""
