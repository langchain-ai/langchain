from typing import Any, Generic

from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import TBaseModel


def strip_think_tags(text: str) -> str:
    """Removes <think>...</think> tags from text.

    Args:
        text: The input text that may contain think tags.
    """

    def remove_think_tags(text: str) -> str:
        """Remove content between <think> and </think> tags more safely."""
        result = []
        i = 0
        while i < len(text):
            # Look for opening tag
            open_tag_pos = text.find("<think>", i)
            if open_tag_pos == -1:
                # No more opening tags, add the rest and break
                result.append(text[i:])
                break

            # Add text before the opening tag
            result.append(text[i:open_tag_pos])

            # Look for closing tag
            close_tag_pos = text.find("</think>", open_tag_pos + 7)
            if close_tag_pos == -1:
                # No closing tag found, treat opening tag as literal text
                result.append("<think>")
                i = open_tag_pos + 7
            else:
                # Skip the content between tags and move past closing tag
                i = close_tag_pos + 8  # "</think>" is 8 characters

        return "".join(result).strip()

    return remove_think_tags(text)


class ReasoningJsonOutputParser(JsonOutputParser):
    """A JSON output parser that strips reasoning tags before parsing.

    This parser removes any content enclosed in <think> tags from the input text
    before delegating to the parent JsonOutputParser for JSON parsing.

    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If `True`, the output will be a JSON object containing
                all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        text = result[0].text
        text = strip_think_tags(text)
        return super().parse_result([Generation(text=text)], partial=partial)


class ReasoningStructuredOutputParser(
    PydanticOutputParser[TBaseModel], Generic[TBaseModel]
):
    """A structured output parser that strips reasoning tags before parsing.

    This parser removes any content enclosed in <think> tags from the input text
    before delegating to the parent PydanticOutputParser for structured parsing.
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a Pydantic object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If `True`, the output will be a JSON object containing
                all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.
        """
        text = result[0].text
        text = strip_think_tags(text)
        return super().parse_result([Generation(text=text)], partial=partial)
