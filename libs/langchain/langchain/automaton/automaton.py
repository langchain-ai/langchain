"""Module containing an automaton definition."""
from __future__ import annotations

import abc
from typing import (
    TypedDict,
    Mapping,
    Any,
    Protocol,
)


class ExecutedState(TypedDict):
    """The response of an action taking LLM."""

    id: str  # the ID of the state that was just executed
    data: Mapping[str, Any]


class State(Protocol):
    """Automaton state protocol."""

    def execute(self) -> ExecutedState:
        """Execute the state, returning the result."""
        ...


class Automaton:
    @abc.abstractmethod
    def get_start_state(self, *args: Any, **kwargs: Any) -> State:
        """Get the start state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_next_state(self, executed_state: ExecutedState) -> State:
        """Get the next state."""
        raise NotImplementedError()
