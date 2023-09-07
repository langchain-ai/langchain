"""Implementation of a think act agent.

Uses priming messages with text based LLM.
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional, Sequence, Union

from langchain.automaton.runnables import (
    create_llm_program,
)
from langchain.automaton.tool_utils import generate_tool_info
from langchain.automaton.typedefs import (
    Agent,
    AgentFinish,
    FunctionCall,
    FunctionResult,
    MessageLike,
    MessageLog,
    PrimingMessage,
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
Respond to the human as helpfully and accurately as 
possible. You have access to the following tools:
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
Use tools if necessary. \
Respond directly if appropriate. \
Format is <action>$BLOB</action> then Observation. \
"""


def generate_memory(tools: Sequence[Tool]) -> MessageLog:
    """Set up basic memory for agent."""
    tools_info = generate_tool_info(tools)

    return MessageLog(
        messages=[
            SystemMessagePromptTemplate.from_template(TEMPLATE_).format(**tools_info)
        ]
    )


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
            data = ast.literal_eval(action_blob)
            name = data["action"]
            if name == "Final Answer":  # Special cased "tool" for final answer
                return AgentFinish(result=data["action_input"])
            return FunctionCall(
                name=data["action"], arguments=data["action_input"] or {}
            )
        else:
            return None


class ThinkActPromptGenerator(PromptValue):
    """think-act paradigm

    prompt value can do additional priming of the LLM, used in to_string method right now.
    """

    message_log: MessageLog

    class Config:
        arbitrary_types_allowed = True

    def to_string(self) -> str:
        """The string variant of the prompt."""
        finalized = []
        messages = self.message_log.messages
        for idx, message in enumerate(messages):
            if isinstance(message, PrimingMessage):
                component = message.content
                finalized.append(component)
                continue

            if isinstance(message, FunctionResult):
                component = f"Observation: {message.result}"
            elif isinstance(message, HumanMessage):
                component = f"Question: {message.content.strip()}"
            elif isinstance(message, (AIMessage, SystemMessage)):
                component = message.content.strip()
            elif isinstance(message, FunctionCall):
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
        for message in self.message_log.messages:
            if isinstance(message, BaseMessage):
                messages.append(message)
            elif isinstance(message, FunctionResult):
                messages.append(
                    SystemMessage(content=f"Observation: `{message.result}`")
                )
        return messages

    @classmethod
    def from_message_log(cls, message_log: MessageLog):
        """Create a ThinkActPromptGenerator from a MessageLog, allowing positional arguments."""
        return cls(message_log=message_log)


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

    def run(self, message_log: MessageLog) -> None:
        """Run the agent."""
        if not message_log:
            raise AssertionError(f"Expected at least one message in message_log")

        for _ in range(self.max_iterations):
            last_message = message_log[-1]

            if isinstance(last_message, AgentFinish):
                break

            # Prime the LLM to start with "Thought: " after an observation
            if isinstance(last_message, (FunctionResult, HumanMessage)):
                message_log.add_messages([PrimingMessage(content="Thought:")])

            message_log.add_messages(self.think_act.invoke(message_log))
