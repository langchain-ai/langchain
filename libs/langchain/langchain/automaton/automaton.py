"""An automaton."""
from __future__ import annotations

import abc
import dataclasses
from typing import (
    Generic,
    TypeVar,
    Sequence,
    Callable,
    Any,
    Union,
    Optional,
    Dict,
    Mapping,
)

from langchain.base_language import BaseLanguageModel
from langchain.prompts import Prompt, BasePromptTemplate


@dataclasses.dataclass(frozen=True)
class Program:
    """A processor that takes in a query and outputs a response."""

    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    parser: Callable[[str], Transition]
    stop: Optional[Sequence[str]] = None

    def execute(self, **kwargs) -> Transition:
        """Execute the program."""
        finalized_prompt = self.prompt.format_prompt(**kwargs)
        llm_output = self.llm.predict(finalized_prompt.to_string(), stop=self.stop)
        return self.parser(llm_output)


@dataclasses.dataclass(frozen=True)
class AbstractState(abc.ABC):
    @abc.abstractmethod
    def execute(self) -> Transition:
        """Execute the state"""


@dataclasses.dataclass(frozen=True)
class LLMState(AbstractState):
    """A state that uses a language model."""

    program: Program
    inputs: Mapping[str, Any]

    def execute(self) -> Transition:
        """Execute the program."""
        return self.program.execute(**self.inputs)


@dataclasses.dataclass(frozen=True)
class UserInputState(AbstractState):
    """Collect user input."""

    def execute(self) -> Transition:
        """Execute the program."""
        user_input = input("Query: ")
        return PayloadTransition(type_="user_input", payload={"question": user_input})


@dataclasses.dataclass(frozen=True)
class EndState(AbstractState):
    """A state that uses a language model."""

    history: History

    def execute(self) -> Transition:
        """Execute the program."""
        raise ValueError()


@dataclasses.dataclass(frozen=True)
class Transition:
    """A transition from one state to another."""


@dataclasses.dataclass(frozen=True)
class PayloadTransition(Transition):
    """A transition from one state to another."""

    type_: str
    payload: Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class LLMTransition(Transition):
    """Any transition out of a program state is a standard transition."""

    llm_output: str
    tool_invocation_request: ToolInvocationRequest


@dataclasses.dataclass(frozen=True)
class History:
    records: Sequence[Union[Transition, AbstractState]] = tuple()

    def append(self, record: Union[Transition, AbstractState]) -> History:
        """Append a record to the history."""
        return dataclasses.replace(self, records=list(self.records) + [record])

    def __repr__(self):
        num_transitions = self.get_num_transitions()
        num_states = self.get_num_states()
        return "History with {} transitions and {} states".format(
            num_transitions, num_states
        )

    def __getitem__(self, item):
        return self.records[item]

    def __len__(self):
        return len(self.records)

    def get_num_transitions(self):
        return sum(1 for record in self.records if isinstance(record, Transition))

    def get_num_states(self):
        return sum(1 for record in self.records if isinstance(record, AbstractState))

    def get_transitions(self) -> Sequence[Transition]:
        return [record for record in self.records if isinstance(record, Transition)]


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


@dataclasses.dataclass(frozen=True)
class Argument:
    """An argument to a tool."""

    name: str
    type_: str
    description: str
    default: Optional[str] = None
    required: bool = False

    def to_string(self) -> str:
        """Get a string representation.

        As a function invocation."""
        return f"{self.name}: {self.type_}, // {self.description}"


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Tool(Generic[T]):
    """A tool."""

    name: str
    description: str
    arguments: Sequence[Argument]
    return_type: str

    callable: Callable[..., Any]

    def to_string(self) -> str:
        """Get a string representation.

        As a function invocation."""
        code = f"type {self.name} = ( // {self.description}\n"

        space = " " * 2

        for arg in self.arguments:
            code += f"{space}{arg.to_string()}\n"

        code += f") => {self.return_type};"
        return code

    def invoke(self, *args, **kwargs) -> T:
        """Invoke the tool."""
        return self.callable(*args, **kwargs)


class Automaton(abc.ABC):
    @abc.abstractmethod
    def get_start_state(self, inputs: Any) -> AbstractState:
        """Get the start state."""

    @abc.abstractmethod
    def get_next_state(
        self, state: AbstractState, transition: Transition, history: History
    ) -> AbstractState:
        """Get the next state."""

    @abc.abstractmethod
    def is_end_state(self, state: AbstractState) -> bool:
        """is the given state an end state."""


class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self, text: str) -> Transition:
        """Parse"""


def _extract_html_tag_content(html: str, tag: str) -> str:
    """Extract all the content between the <tag>CONTENT</tag>."""
    # TODO(Replace with actual solution based on bs4)
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    try:
        start_index = html.index(start_tag) + len(start_tag)
        end_index = html.index(end_tag)
        return html[start_index:end_index]
    except ValueError:
        return f'respond("{html}")'


def _get_content_between_parenthesis(text: str) -> str:
    """Get the content between the first pair of parenthesis."""
    start_index = text.index("(") + 1
    end_index = text.index(")")
    return text[start_index:end_index]


class ChatParser(Parser):
    def parse(self, text: str) -> Transition:
        act = _extract_html_tag_content(text, "act")
        if act.startswith("search"):
            name = "search"
            argument = _get_content_between_parenthesis(act)
            tool_invocation_request = ToolInvocationRequest(
                name=name, arguments=[argument]
            )
        elif act.startswith("respond"):
            name = "respond"
            argument = _get_content_between_parenthesis(act)
            tool_invocation_request = ToolInvocationRequest(
                name=name, arguments=[argument]
            )
        else:
            tool_invocation_request = None

        return LLMTransition(
            llm_output=text.strip(),
            tool_invocation_request=tool_invocation_request,
        )


@dataclasses.dataclass
class ToolInvocationRequest:
    name: str
    arguments: Sequence[Any]


def parse_function_invocation(text: str) -> ToolInvocationRequest:
    """Parse a function invocation.

    Parse an invocation like:

    `foo(1, 2)`

    Into:

    ToolInvocation(name="foo", arguments=[1, 2])
    """
    return ToolInvocationRequest(name="search", arguments=[text])


@dataclasses.dataclass(frozen=True)
class ToolRegistry:
    """A registry of tools."""

    tools: Mapping[str, Tool]

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        return self.tools[name]

    def as_string(self) -> str:
        """Get a string representation of the tool registry."""
        return "\n".join(tool.to_string() for tool in self.tools.values())


NEW_AGE_PROMPT = """\
You are an AI assistant. Do not reveal any other information about yourself. \
Your goal is to help the user to the best of your ability.

You may invoke any of the tools to help achieve your objective.

Do not guess the answer. Rely on the contents of the `Knowledge` as it has been independently
verified and is known to be correct.

If the question is answerable using the Knowledge, then use the `respond` tool to answer. \
Otherwise, use other tools to get more knowledge. \

Tools:

{tools}

---

Examples:

Question: Who was the first president of the United States?
Knowledge: ```The first president of the United States was Ella.```
Thought: The question is answerable using current Knowledge.
Action: <act>respond("Ella")</act>

Question: What are the colors on the building in front of me?
Knowledge: ```The user's name is Eugene.```
Thought: The question is not answerable using current Knowledge or any tools.
Action: <act>respond("I do not have enough information to answer that question.")</act>

Question: Wehqo Qohwefwqej
Knowledge: ```The user is standing outside of a red and blue building.```
Thought: The question does not make sense.
Action: <act>respond("I am not sure what you mean.")</act>

Question: Hello!
Knowledge: ``````
Thought: The user is greeting me.
Action: <act>respond("Hello! How can I help you?")</act>

---

Question: {question}
Knowledge: ```{knowledge}```
Thought:\
"""

TOOL_REGISTRY = ToolRegistry(
    tools={
        "search": Tool(
            name="search",
            description="Use to search the internet",
            arguments=[
                Argument(
                    name="query",
                    type_="string",
                    description="What to search for",
                )
            ],
            return_type="string",
            callable=lambda query: "search result",
        ),
        "respond": Tool(
            name="respond",
            description="Response to the user.",
            arguments=[
                Argument(
                    name="response",
                    type_="string",
                    description="The response to the user.",
                )
            ],
            return_type="null",
            callable=lambda query: "responding to user.",
        ),
    },
)


@dataclasses.dataclass(frozen=True)
class AbstractMemory:
    """Abstract memory."""


@dataclasses.dataclass(frozen=True)
class RunningMemory(AbstractMemory):
    """Running memory."""

    messages: Sequence[str] = ()

    def add_message(self, message: str) -> RunningMemory:
        """Add a message to the memory."""
        return dataclasses.replace(self, messages=list(self.messages) + [message])

    def to_string(self) -> str:
        """Get a string representation."""
        return "\n".join(self.messages)


class ChatAutomaton(Automaton):
    def __init__(self, llm: BaseLanguageModel, prompt: Prompt) -> None:
        self.llm = llm
        self.tool_registry = TOOL_REGISTRY
        self.prompt = prompt.partial(tools=self.tool_registry.as_string())
        self.stop = ["Question:", "Knowledge:", "Thought:"]
        self.parser = ChatParser()

    def get_start_state(self, inputs: Mapping[str, Any]) -> AbstractState:
        """Get the start state."""
        return self.get_user_input_state(inputs=inputs)

    def get_llm_state(self, inputs: Mapping[str, Any]) -> AbstractState:
        program = Program(
            prompt=self.prompt, llm=self.llm, parser=self.parser.parse, stop=self.stop
        )
        return LLMState(program=program, inputs=inputs)

    def get_user_input_state(self, inputs: Mapping[str, Any]) -> AbstractState:
        """Get the start state."""
        return UserInputState()

    def get_next_state(
        self, state: AbstractState, transition: Transition, history: History
    ) -> AbstractState:
        if history.get_num_states() >= 10:
            return EndState(history=history)
        if isinstance(state, UserInputState) and isinstance(
            transition, PayloadTransition
        ):
            transitions = history.get_transitions()

            memory = []

            for transition in transitions:
                if isinstance(transition, LLMTransition):
                    if transition.tool_invocation_request.name == "respond":
                        memory.append(f"AI: {transition.llm_output}")
                elif isinstance(transition, PayloadTransition):
                    memory.append(f"User: {transition.payload['question']}")
                else:
                    raise AssertionError()

            memory_str = "\n".join(memory)

            inputs = {
                "question": transition.payload["question"],
                "knowledge": memory_str,
            }
            return self.get_llm_state(inputs=inputs)
        else:
            if isinstance(transition, LLMTransition):
                tool_invocation = transition.tool_invocation_request

                if tool_invocation.name == "respond":
                    return UserInputState()
                else:
                    return self.get_llm_state(inputs=state.inputs)
            else:
                raise AssertionError()

    def is_end_state(self, state: AbstractState) -> bool:
        """is the given state an end state."""
        return isinstance(state, EndState)
