from langchain_core.agents import AgentAction

from langchain.agents.conversational.output_parser import ConvoOutputParser


def test_normal_output_parsing() -> None:
    _test_convo_output(
        """
Action: my_action
Action Input: my action input
""",
        "my_action",
        "my action input",
    )


def test_multiline_output_parsing() -> None:
    _test_convo_output(
        """
Thought: Do I need to use a tool? Yes
Action: evaluate_code
Action Input: Evaluate Code with the following Python content:
```python
print("Hello fifty shades of gray mans!"[::-1])
```
""",
        "evaluate_code",
        """
Evaluate Code with the following Python content:
```python
print("Hello fifty shades of gray mans!"[::-1])
```""".lstrip(),
    )


def _test_convo_output(
    input: str, expected_tool: str, expected_tool_input: str
) -> None:
    result = ConvoOutputParser().parse(input.strip())
    assert isinstance(result, AgentAction)
    assert result.tool == expected_tool
    assert result.tool_input == expected_tool_input
