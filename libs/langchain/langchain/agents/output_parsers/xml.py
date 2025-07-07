import re
import xml.sax.saxutils
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from pydantic import Field

from langchain.agents import AgentOutputParser


class XMLAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers from XML-formatted agent output.

    This parser is hardened against XML injection by using standard XML entity
    decoding for content within tags. It is designed to work with the corresponding
    ``format_xml`` function.

    Args:
        unescape_xml: If True, the parser will unescape XML special characters in the
            content of tags. This should be enabled if the agent's output was formatted
            with XML escaping.

            If False, the parser will return the raw content as is, which may include
            XML special characters like `<`, `>`, and `&`.

    Expected formats:
        Tool invocation (returns AgentAction):
            <tool>search</tool>
            <tool_input>what is 2 + 2</tool_input>

        Final answer (returns AgentFinish):
            <final_answer>The answer is 4</final_answer>

    Raises:
        ValueError: If the input doesn't match either expected XML format or
            contains malformed XML structure.
    """

    unescape_xml: bool = Field(default=True)
    """If True, the parser will unescape XML special characters in the content
    of tags. This should be enabled if the agent's output was formatted
    with XML escaping.

    If False, the parser will return the raw content as is,
    which may include XML special characters like `<`, `>`, and `&`.
    """

    def _extract_tag_content(
        self, tag: str, text: str, *, required: bool
    ) -> Union[str, None]:
        """
        Extracts content from a specified XML tag, ensuring it appears at most once.

        Args:
            tag: The name of the XML tag (e.g., ``'tool'``).
            text: The text to parse.
            required: If True, a ValueError will be raised if the tag is not found.

        Returns:
            The unescaped content of the tag as a string, or None if not found
            and not required.

        Raises:
            ValueError: If the tag appears more than once, or if it is required
                but not found.
        """
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)

        if len(matches) > 1:
            raise ValueError(
                f"Malformed XML: Found {len(matches)} <{tag}> blocks. Expected 0 or 1."
            )

        if not matches:
            if required:
                raise ValueError(f"Malformed XML: Missing required <{tag}> block.")
            return None

        content = matches[0]
        if self.unescape_xml:
            entities = {"&apos;": "'", "&quot;": '"'}
            return xml.sax.saxutils.unescape(content, entities)
        return content

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parses the given text into an AgentAction or AgentFinish object.
        """
        # Check for a tool invocation
        if "<tool>" in text and "</tool>" in text:
            tool = self._extract_tag_content("tool", text, required=True)
            if tool is None:
                raise ValueError("Tool content should not be None when required=True")
            # Tool input is optional
            tool_input = (
                self._extract_tag_content("tool_input", text, required=False) or ""
            )

            return AgentAction(tool=tool, tool_input=tool_input, log=text)

        # Check for a final answer
        elif "<final_answer>" in text and "</final_answer>" in text:
            answer = self._extract_tag_content("final_answer", text, required=True)
            return AgentFinish(return_values={"output": answer}, log=text)

        # If neither format is found, raise an error
        else:
            raise ValueError(
                "Could not parse LLM output. Expected a tool invocation with <tool> "
                "and <tool_input> tags, or a final answer with <final_answer> tags."
            )

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-agent"
