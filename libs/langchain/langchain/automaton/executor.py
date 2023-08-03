from __future__ import annotations

from typing import Any

from langchain.automaton.automaton import Automaton, EndState, LLMTransition
from langchain.automaton.history import History


class Executor:
    def __init__(self, automaton: Automaton, debug: bool = False) -> None:
        """Initialize the executor."""
        self.automaton = automaton
        self.debug = debug

    def execute(self, inputs: Any) -> EndState:
        """Execute the query."""
        state = self.automaton.get_start_state(inputs)
        history = History()
        i = 0
        while True:
            history = history.append(state)
            transition = state.execute()
            if self.debug:
                if isinstance(transition, LLMTransition):
                    print("AI:", transition.llm_output)
            history = history.append(transition)
            next_state = self.automaton.get_next_state(state, transition, history)
            state = next_state
            i += 1
            if self.automaton.is_end_state(state):
                return state

            if i > 100:
                return EndState(history=history)
