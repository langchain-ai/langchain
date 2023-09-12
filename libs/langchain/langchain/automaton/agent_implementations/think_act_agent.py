"""Implementation of a think act agent.

Uses priming messages with text based LLM.
"""
from __future__ import annotations

import ast
import re
from typing import Iterator, List, Optional, Sequence, Union

from langchain.automaton.runnables import (
    create_llm_program,
)
from langchain.automaton.tool_utils import generate_tool_info
from langchain.automaton.typedefs import (
    AdHocMessage,
    Agent,
    AgentFinish,
    FunctionCallRequest,
    FunctionCallResponse,
    MessageLike,
)
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    PromptValue,
    SystemMessage,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool, Tool

TEMPLATE_ = """\
Respond to the human as helpfully and accurately as possible. \
You have access to the following tools:
{tools_description}
Use a blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

You can only use a single tool at a time.

<action>
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
</action>

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
<action>
$BLOB
</action>

Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
<action>
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
</action>

Begin:

Reminder to ALWAYS respond with a valid blob of a single action. \
Use tools if necessary. Respond directly if appropriate. \
Format is <action>$BLOB</action> then Observation.\
"""


def generate_memory(tools: Sequence[Tool]) -> List[MessageLike]:
    """Set up basic memory for agent."""
    tools_info = generate_tool_info(tools)
    return [SystemMessagePromptTemplate.from_template(TEMPLATE_).format(**tools_info)]


class ActionParser:
    """A utility class to encode and decode action blocks."""

    def __init__(self) -> None:
        """Initialize the ActionParser."""
        self.pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)

    def decode(self, text: Union[BaseMessage, str]) -> Optional[MessageLike]:
        """Decode the action."""
        if isinstance(text, BaseMessage):
            text = text.content
        match = self.pattern.search(text)
        if match:
            action_blob = match.group("action_blob")
            try:
                data = ast.literal_eval(action_blob)
            except SyntaxError:
                return AdHocMessage(
                    type="error", data=f"Invalid action blob {action_blob}"
                )
            name = data["action"]

            if name == "Final Answer":  # Special cased "tool" for final answer
                return AgentFinish(result=data["action_input"])
            action_input = data["action_input"]
            if isinstance(action_input, str) and not action_input:
                named_arguments = {}
            elif isinstance(action_input, dict):
                named_arguments = action_input
            else:
                return AdHocMessage(
                    type="error",
                    data=f"Invalid action blob {action_blob}, action_input must be a dict",
                )

            return FunctionCallRequest(
                name=data["action"], named_arguments=named_arguments or {}
            )
        else:
            return None


class ThinkActPromptGenerator(PromptValue):
    """think-act paradigm

    prompt value can do additional priming of the LLM, used in to_string method right now.
    """

    messages: Sequence[MessageLike]

    class Config:
        arbitrary_types_allowed = True

    def to_string(self) -> str:
        """The string variant of the prompt."""
        finalized = []
        messages = self.messages
        for idx, message in enumerate(messages):
            if isinstance(message, AdHocMessage):
                if message.type == "error":
                    finalized.extend(
                        [
                            f"Error: Malformed <action> blob with error: {message.data}. "
                            f"Please re-write the action correctly."
                            "\n",
                        ]
                    )
                    continue
                if message.type != "prime":
                    raise AssertionError()
                component = message.data
                finalized.append(component)
                continue

            if isinstance(message, FunctionCallResponse):
                component = f"Observation: {message.result}"
            elif isinstance(message, HumanMessage):
                component = f"Question: {message.content.strip()}"
            elif isinstance(message, (AIMessage, SystemMessage)):
                component = message.content.strip()
            elif isinstance(message, FunctionCallRequest):
                # This is an internal message, and should not be returned to the user.
                continue
            elif isinstance(message, AgentFinish):
                component = f"Answer: {message.result}"
            else:
                raise NotImplementedError()
            finalized.extend([component, "\n"])
        return "".join(finalized).strip()

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        messages = []
        for message in self.messages:
            if isinstance(message, BaseMessage):
                messages.append(message)
            elif isinstance(message, FunctionCallResponse):
                messages.append(
                    SystemMessage(content=f"Observation: `{message.result}`")
                )
        return messages

    @classmethod
    def from_message_log(cls, messages: Sequence[MessageLike]):
        """Create a ThinkActPromptGenerator from a MessageLog, allowing positional arguments."""
        return cls(messages=messages)


class ThinkActAgent(Agent):
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        *,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the chat automaton."""
        self.think_act = create_llm_program(
            llm,
            prompt_generator=ThinkActPromptGenerator.from_message_log,
            stop=["Observation:", "observation:"],
            parser=ActionParser().decode,
            tools=tools,
            invoke_tools=True,
        )
        self.max_iterations = max_iterations

    def run(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[dict] = None,
        max_iterations: int = 100,
    ) -> Iterator[MessageLike]:
        """Run the agent."""
        all_messages = list(messages)
        for _ in range(max_iterations):
            if all_messages and isinstance(all_messages[-1], AgentFinish):
                break

            if all_messages and isinstance(
                all_messages[-1], (FunctionCallResponse, HumanMessage)
            ):
                all_messages.append(AdHocMessage(type="prime", data="Thought:"))

            new_messages = self.think_act.invoke(all_messages, config=config)
            yield from new_messages
            all_messages.extend(new_messages)
