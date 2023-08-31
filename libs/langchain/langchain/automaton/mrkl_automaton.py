from __future__ import annotations

import ast
import re
from typing import Any, Sequence, Optional, Union, Dict

from langchain.automaton.automaton import Automaton
from langchain.automaton import typedefs
from langchain.automaton.typedefs import (
    MessageLog,
    MessageLogPromptValue,
)
from langchain.automaton.well_known_states import (
    create_llm_program,
    create_tool_invoker,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import PromptValue, BaseMessage
from langchain.schema.language_model import BaseLanguageModel
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

Reminder to ALWAYS respond with a valid json blob of a single action. \
Use tools if necessary. \
Respond directly if appropriate. \
Format is <action>$BLOB</action> then Observation. \
"""


def _generate_prompt(message_log: MessageLog) -> PromptValue:
    """Generate prompt value."""
    return MessageLogPromptValue(message_log=message_log)


def _generate_mrkl_memory(tools: Sequence[Tool]) -> MessageLog:
    """Set up memory to act as a MRKL agent."""
    tool_strings = (
        "\n".join([f"{tool_.name}: {tool_.description}" for tool_ in tools]) + "\n"
    )
    tool_names = ", ".join([tool_.name for tool_ in tools])
    chat_prompt_template = ChatPromptTemplate.from_messages([("system", TEMPLATE_)])
    return MessageLog(
        messages=chat_prompt_template.format_messages(
            tools=tool_strings, tool_names=tool_names
        )
    )


class ActionParser:
    """A utility class to encode and decode action blocks."""

    def __init__(self) -> None:
        """Initialize the ActionParser."""
        self.pattern = re.compile(r"<action>(?P<action_blob>.*?)<\/action>", re.DOTALL)

    def decode(self, text: Union[BaseMessage, str]) -> Optional[Dict[str, Any]]:
        """Decode the action."""
        if isinstance(text, BaseMessage):
            text = text.content
        match = self.pattern.search(text)
        if match:
            action_blob = match.group("action_blob")
            action = ast.literal_eval(action_blob)
            return {
                "type": "function_call",
                "data": {
                    "name": action["action"],
                    "inputs": action["action_input"] or {},
                },
            }
        else:
            return None


class MRKLAutomaton(Automaton):
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
    ) -> None:
        """Initialize the chat automaton."""
        super().__init__()
        self.think_act = create_llm_program(
            llm,
            prompt_generator=_generate_prompt,
            stop=["Observation"],
            parser=ActionParser().decode,
        )
        self.tool_invoker = create_tool_invoker(tools)

    def run(self, message_log: MessageLog) -> None:
        """Run the agent."""
        if not message_log.messages:
            raise AssertionError()
        last_message = message_log.messages[-1]

        # New messages
        messages = []
        if isinstance(last_message, typedefs.FunctionCall):
            messages.append(self.tool_invoker.invoke(last_message))
        else:
            messages = self.think_act.invoke(message_log)

        message_log.add_messages(messages)
