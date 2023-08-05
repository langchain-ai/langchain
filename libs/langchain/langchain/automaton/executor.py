from __future__ import annotations

from typing import Any

from langchain.automaton.automaton import Automaton, EndState, LLMTransition
from langchain.automaton.history import History
from langchain.schema import SystemMessage, AIMessage


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

#
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant cat. Please use the tools at your disposal to help the human. You can ask the user to clarify if need be.",
        ),
    ]
)

last_response = None


class SingleAgentExecutor:
    pass


while True:
    last_message = template.format_messages()[-1]
    message_printer(last_message)
    print("---")

    if last_response and last_response["name"] == "bye":
        print("AGI: byebye silly human")
        break

    # Very hacky routing layer
    if isinstance(last_message, SystemMessage) or (
        (last_message, AIMessage) and not last_message.additional_kwargs
    ):  # (Ready for human input?)
        # Determine if human turn
        content = input("User:")
        if content == "q":  # Quit
            break
        template = template + [("human", content)]
    else:  # Determine if need to insert function invocation information
        if (
            last_response and last_response["name"]
        ):  # Last response was a tool invocation, need to append a Function message
            template = template + [
                FunctionMessage(
                    content=last_response["data"], name=last_response["name"]
                )
            ]
            message_printer(template.messages[-1])

    last_response = chain.invoke(
        template.format_messages()
    )  # would love to get rid of format
    template = template + last_response["message"]
