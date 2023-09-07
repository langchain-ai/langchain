from __future__ import annotations

import langchain.automaton.agent_implementations.xml_agent
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.automaton.mrkl_agent import ActionParser


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


def test_parser() -> None:
    """Tes the parser."""
    sample_text = """
    Some text before
    <action>
    {
      "key": "value",
      "number": 42
    }
    </action>
    Some text after
    """
    action_parser = ActionParser()
    action = langchain.automaton.agent_implementations.xml_agent.decode(sample_text)
    assert action == {
        "key": "value",
        "number": 42,
    }
