from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents import AgentOutputParser


class XMLAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in XML format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    <tool>search</tool>
    <tool_input>what is 2 + 2</tool_input>
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    <final_answer>Foo</final_answer>
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            if "<tool_input>" in tool_input:
                _tool_input = tool_input.split("<tool_input>")[1]
                if "</tool_input>" in _tool_input:
                    _tool_input = _tool_input.split("</tool_input>")[0]
            else:
                raise OutputParserException(
                    error=ValueError("Invalid format for output."),
                    llm_output=text,
                    observation=(
                        "ERROR: For a fool invocation, be sure to include a <tool_input> and"
                        "</tool_input> tags. A function without parameters could be invoked with a "
                        "an empty dictionary as the tool input.\n"
                        "To invoke a tool, use the format "
                        "`<tool>$TOOL_NAME</tool><tool_input>$TOOL_INPUT</tool_input>`.\n "
                    ),
                )
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            if "</final_answer>" in answer:
                answer = answer.split("</final_answer>")[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            raise OutputParserException(
                error=ValueError("Invalid format for output."),
                llm_output=text,
                observation=(
                    "ERROR: Please either invoke a tool or provide a final answer."
                    "To invoke a tool, use the format "
                    "`<tool>$TOOL_NAME</tool><tool_input>$TOOL_INPUT</tool_input>`. "
                    "where $TOOL_NAME is one of the provided tools and $TOOL_INPUT "
                    "is a dictionary of arguments to pass to the tool, "
                    "matching the schema.\n"
                ),
                send_to_llm=True,
            )

    def get_format_instructions(self) -> str:
        """Get the format instructions for this output parser."""
        raise NotImplementedError(
            "XMLAgentOutputParser does contain format instructions."
        )

    @property
    def _type(self) -> str:
        return "xml-agent"
