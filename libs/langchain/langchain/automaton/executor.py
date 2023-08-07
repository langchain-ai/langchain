from __future__ import annotations

from typing import (
    List,
    Tuple,
)

from langchain.automaton.automaton import ExecutedState, State, Automaton
from langchain.automaton.typedefs import Memory


# Need to make into runnable
# This is a for looping runnable... :)
class Executor:
    def __init__(
        self, automaton: Automaton, memory: Memory, max_iterations: int
    ) -> None:
        """Initialize the executor."""
        self.automaton = automaton
        self.max_iterations = max_iterations
        self.memory = memory

    def run(self) -> Tuple[State, List[ExecutedState]]:
        """Run the automaton.

        Returns:
            The final state and result of executed states.
        """
        state = self.automaton.get_start_state()
        executed_states = []

        for _ in range(self.max_iterations):
            executed_state = state.execute(memory=self.memory)
            executed_states.append(executed_state)
            # Should the transition function get memory?
            state = self.automaton.get_next_state(executed_state)

        return state, executed_states
