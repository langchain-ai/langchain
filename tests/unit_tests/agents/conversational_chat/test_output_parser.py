from __future__ import annotations

import pytest

from langchain.agents.conversational_chat.output_parser import (
    ConvoOutputParser,
    OutputParserException,
)
from langchain.schema import AgentAction, AgentFinish


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            '```json\n'
            '{"action": "Final Answer", "action_input": "Hello, World!"}'
            '```',
            AgentFinish(
                {"output": "Hello, World!"},
                '```json\n'
                '{"action": "Final Answer", "action_input": "Hello, World!"}'
                '```'
            )
        ),
        (
            '```json\n'
            '{"action": "Some Action", "action_input": "Hello, World!"}'
            '```', 
            AgentAction(
                "Some Action",
                "Hello, World!",
                '```json\n'
                '{"action": "Some Action", "action_input": "Hello, World!"}'
                '```'
            )
        ),
        (
            '```json\n'
            '{"action": "Final Answer", "action_input": '
            '"Here\'s a simple Python \'Hello World!\' program:\\n\\n'
            'python\\nprint(\'Hello World!\')\\n"}'
            '```',
            AgentFinish(
                {
                    "output": (
                        "Here's a simple Python 'Hello World!' program:\n\n"
                        "python\nprint('Hello World!')\n"
                    )
                },
                '```json\n'
                '{"action": "Final Answer", "action_input": '
                '"Here\'s a simple Python \'Hello World!\' program:\\n\\n'
                'python\\nprint(\'Hello World!\')\\n"}'
                '```'
            )
        ),
    ]
)
def test_parser(text, expected) -> None:
    """Test the ConvoOutputParser class."""
    parser = ConvoOutputParser()
    assert parser.parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        ('Invalid text without any recognizable format'),
        ('json\n{"invalid": "json"}'),
        ('```json\n{"invalid": "json"}```'),
    ]
)
def test_unhappy_path(text) -> None:
    """Test the ConvoOutputParser class."""
    parser = ConvoOutputParser()
    with pytest.raises(OutputParserException):
        parser.parse(text)

