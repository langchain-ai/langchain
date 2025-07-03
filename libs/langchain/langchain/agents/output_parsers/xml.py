import re
from typing import Literal, Union

from langchain_core.agents import AgentAction, AgentFinish
from pydantic import Field

from langchain.agents import AgentOutputParser


def _unescape(text: str) -> str:
    """Convert custom tag delimiters back into XML tags."""
    replacements = {
        "[[tool]]": "<tool>",
        "[[/tool]]": "</tool>",
        "[[tool_input]]": "<tool_input>",
        "[[/tool_input]]": "</tool_input>",
        "[[observation]]": "<observation>",
        "[[/observation]]": "</observation>",
    }
    for repl, orig in replacements.items():
        text = text.replace(repl, orig)
    return text


class XMLAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers from XML-formatted agent output.

    This parser extracts structured information from XML tags to determine whether
    an agent should perform a tool action or provide a final answer. It includes
    built-in escaping support to safely handle tool names and inputs
    containing XML special characters.

    Args:
        escape_format: The escaping format to use when parsing XML content.
            Supports 'minimal' which uses custom delimiters like [[tool]] to replace
            XML tags within content, preventing parsing conflicts.
            Use 'minimal' if using a corresponding encoding format that uses
            the _escape function when formatting the output (e.g., with format_xml).

    Expected formats:
        Tool invocation (returns AgentAction):
            <tool>search</tool>
            <tool_input>what is 2 + 2</tool_input>

        Final answer (returns AgentFinish):
            <final_answer>The answer is 4</final_answer>

    Note:
        Minimal escaping allows tool names containing XML tags to be safely
        represented. For example, a tool named "search<tool>nested</tool>" would be
        escaped as "search[[tool]]nested[[/tool]]" in the XML and automatically
        unescaped during parsing.

    Raises:
        ValueError: If the input doesn't match either expected XML format or
            contains malformed XML structure.
    """

    escape_format: Literal["minimal"] | None = Field(default="minimal")
    """The format to use for escaping XML characters.
    
    minimal - uses custom delimiters to replace XML tags within content,
    preventing parsing conflicts. This is the only supported format currently.
    
    None - no escaping is applied, which may lead to parsing conflicts.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            if "</tool_input>" in _tool_input:
                _tool_input = _tool_input.split("</tool_input>")[0]
            # Unescape custom delimiters in tool name and input
            if self.escape_format == "minimal":
                _tool = _unescape(_tool)
                _tool_input = _unescape(_tool_input)
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text and "</final_answer>" in text:
            matches = re.findall(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
            if len(matches) != 1:
                msg = (
                    "Malformed output: expected exactly one "
                    "<final_answer>...</final_answer> block."
                )
                raise ValueError(msg)
            answer = matches[0]
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            msg = (
                "Malformed output: expected either a tool invocation "
                "or a final answer in XML format."
            )
            raise ValueError(msg)

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"
