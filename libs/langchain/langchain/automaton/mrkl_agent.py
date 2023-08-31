from __future__ import annotations

import ast
import re
from typing import Sequence, Optional, Union, List

from langchain.automaton import typedefs
from langchain.automaton.runnables import (
    create_llm_program,
    create_tool_invoker,
)
from langchain.automaton.typedefs import (
    MessageLog,
    MessageLike,
    FunctionCall,
    FunctionResult,
)
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import (
    PromptValue,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool, Tool

TEMPLATE_ = """\
Respond to the human as helpfully and accurately as 
possible. You have access to the following tools:
{tools}
Use a blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $BLOB, as shown:

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


def _generate_tools_descriptions(tools: Sequence[Tool]) -> str:
    """Generate a description of the tools."""
    return "\n".join([f"{tool_.name}: {tool_.description}" for tool_ in tools]) + "\n"


def generate_mrkl_memory(tools: Sequence[Tool]) -> MessageLog:
    """Set up memory to act as a MRKL agent."""
    tools_description = _generate_tools_descriptions(tools)
    tool_names = ", ".join([tool_.name for tool_ in tools])
    return MessageLog(
        messages=[
            SystemMessagePromptTemplate.from_template(TEMPLATE_).format(
                tools=tools_description, tool_names=tool_names
            )
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
            if name == "Final Answer":
                return typedefs.AgentFinish(result=data["action_input"])
            return typedefs.FunctionCall(
                name=data["action"], arguments=data["action_input"] or {}
            )
        else:
            return None


class ThinkActPromptGenerator(PromptValue):
    """use think act paradaigm.

    prompt value can do additional priming of the LLM, used in to_string method right now.
    """

    message_log: MessageLog

    class Config:
        arbitrary_types_allowed = True

    def to_string(self) -> str:
        """The string variant of the prompt."""
        finalized = []
        messages = self.to_messages()
        for message in messages:
            if isinstance(message, FunctionResult):
                component = f"Observation: {message.result}"
            elif isinstance(
                message, FunctionCall
            ):  # Used LLMs that do not support a FunctionCallMessage
                # This is an internal message, and should not be returned to the user.
                continue
            elif isinstance(message, HumanMessage):
                component = f"Question: {message.content}"
            elif isinstance(message, (AIMessage, SystemMessage)):
                component = message.content
            else:
                raise NotImplementedError()

            finalized.append(component)

        if messages and isinstance(messages[-1], (HumanMessage, FunctionCall)):
            # Prime the AI to think
            finalized.append("Thought:")

        return "\n".join(finalized)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        return [
            message
            for message in self.message_log.messages
            if isinstance(message, BaseMessage)
        ]

    @classmethod
    def from_message_log(cls, message_log: MessageLog):
        """Create a ThinkActPromptGenerator from a MessageLog, allowing positional arguments."""
        return cls(message_log=message_log)


class MRKLAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
    ) -> None:
        """Initialize the chat automaton."""
        self.think_act: Runnable[
            MessageLog, Sequence[MessageLike]
        ] = create_llm_program(
            llm,
            prompt_generator=ThinkActPromptGenerator.from_message_log,
            stop=["Observation:", "observation:"],
            parser=ActionParser().decode,
        )
        self.tool_invoker: Runnable[FunctionCall, FunctionResult] = create_tool_invoker(
            tools
        )

    def run(self, message_log: MessageLog) -> None:
        """Run the agent."""
        if not message_log.messages:
            raise AssertionError()

        last_message = message_log.messages[-1]

        max_iterations = 10
        iteration_num = 0

        while True:
            if iteration_num > max_iterations:
                break

            match last_message:
                case typedefs.AgentFinish:
                    break
                case typedefs.FunctionCall:
                    messages = self.tool_invoker.invoke(last_message)
                case _:
                    messages = self.think_act.invoke(message_log)

            if not messages:
                raise AssertionError(f"No messages returned from last step")

            message_log.add_messages(messages)
            last_message = messages[-1]
            iteration_num += 1
