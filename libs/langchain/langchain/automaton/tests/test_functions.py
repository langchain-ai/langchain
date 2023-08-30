from __future__ import annotations

import re
from typing import List, Sequence

from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.automaton.open_ai_functions import (
    OpenAIFunctionsRouter,
    SimpleChatGenerator,
)
from langchain.automaton.tests.utils import FakeChatOpenAI
from langchain.automaton.typedefs import Memory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import PromptValue, AIMessage
from langchain.schema.runnable import RunnableLambda
from langchain.tools import tool, Tool


def get_tools() -> List[Tool]:
    @tool
    def name() -> str:
        """Use to look up the user's name"""
        return "Eugene"

    @tool
    def get_weather(city: str) -> str:
        """Get weather in a specific city."""
        return "42F and sunny"

    @tool
    def add(x: int, y: int) -> int:
        """Use to add two numbers."""
        return x + y

    return list(locals().values())


TEMPLATE_ = """\
Respond to the human as helpfully and accurately as 
possible. You have access to the following tools:
{tools}
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

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
$JSON_BLOB
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
Format is <action>$JSON_BLOB</action> then Observation\
"""


class PromptGenerator:
    def __init__(self, tools: Sequence[Tool]) -> None:
        """Prompt generation interface."""
        self.tools = tools

    def generate(self, memory: Memory) -> PromptValue:
        """Generate a prompt value from memory."""
        tools = get_tools()
        tool_strings = (
            "\n".join([f"{tool_.name}: {tool_.description}" for tool_ in tools]) + "\n"
        )
        tool_names = ", ".join([tool_.name for tool_ in tools])
        chat_prompt_template = ChatPromptTemplate.from_messages([("system", TEMPLATE_)])
        return SimpleChatGenerator(
            messages=chat_prompt_template.format_messages(
                tools=tool_strings, tool_names=tool_names
            )
        )


def test_structured_output_chat() -> None:
    parser = StructuredChatOutputParser()
    output = parser.parse(
        """
        ```json
        {
            "action": "hello",
            "action_input": {
                "a": 2
            }
        }
        ```
        """
    )
    assert output == {}


def test_openai_functions_router() -> None:
    """Test the OpenAIFunctionsRouter."""

    def revise(notes: str) -> str:
        """Revises the draft."""
        return f"Revised draft: {notes}!"

    def accept(draft: str) -> str:
        """Accepts the draft."""
        return f"Accepted draft: {draft}!"

    router = OpenAIFunctionsRouter(
        functions=[
            {
                "name": "revise",
                "description": "Sends the draft for revision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "The editor's notes to guide the revision.",
                        },
                    },
                },
            },
            {
                "name": "accept",
                "description": "Accepts the draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "string",
                            "description": "The draft to accept.",
                        },
                    },
                },
            },
        ],
        runnables={
            "revise": RunnableLambda(lambda x: revise(x["revise"])),
            "accept": RunnableLambda(lambda x: accept(x["draft"])),
        },
    )

    model = FakeChatOpenAI(
        message_iter=iter(
            [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "accept",
                            "arguments": '{\n  "draft": "turtles"\n}',
                        }
                    },
                )
            ]
        )
    )

    chain = model.bind(functions=router.functions) | router

    assert chain.invoke("Something about turtles?") == "Accepted draft: turtles!"
