from langchain.agents.output_parsers.json import JSONAgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish


def test_tool_usage() -> None:
    parser = JSONAgentOutputParser()
    _input = """    ```
{
  "action": "search",
  "action_input": "2+2"
}
```"""
    output = parser.invoke(_input)
    expected_output = AgentAction(tool="search", tool_input="2+2", log=_input)
    assert output == expected_output


def test_finish() -> None:
    parser = JSONAgentOutputParser()
    _input = """```
{
  "action": "Final Answer",
  "action_input": "4"
}
```"""
    output = parser.invoke(_input)
    expected_output = AgentFinish(return_values={"output": "4"}, log=_input)
    assert output == expected_output
