from langchain.agents.conversational.output_parser import ConvoOutputParser


def test_normal_output_parsing():
    result = ConvoOutputParser().parse(
        """
Action: my_action
Action Input: my action input
""".strip()
    )
    assert result.tool == "my_action"
    assert result.tool_input == "my action input"


def test_multiline_output_parsing():
    result = ConvoOutputParser().parse(
        """
Thought: Do I need to use a tool? Yes
Action: evaluate_code
Action Input: Evaluate Code with the following Python content:
```python
print("Hello fifty shades of gray mans!"[::-1])
```
""".strip()
    )
    assert result.tool == "evaluate_code"
    assert (
        result.tool_input
        == """
Evaluate Code with the following Python content:
```python
print("Hello fifty shades of gray mans!"[::-1])
```""".lstrip()
    )


def test_output_parsing_with_observation():
    result = ConvoOutputParser().parse(
        """
Thought: Do I need to use a tool? Yes
Action: my_tool
Action Input: the input to the action
Observation: the result of the action
""".strip()
    )
    assert result.tool == "my_tool"
    assert result.tool_input == "the input to the action"


def test_output_parsing_with_partial_observation():
    result = ConvoOutputParser().parse(
        """
Thought: Do I need to use a tool? Yes
Action: my_tool
Action Input: the input to the action
Obs""".strip()
    )
    assert result.tool == "my_tool"
    assert result.tool_input == "the input to the action"
