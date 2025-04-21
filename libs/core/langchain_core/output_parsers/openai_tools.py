"""Parse tools for OpenAI tools output."""

import copy
import json
import logging
from json import JSONDecodeError
from typing import Annotated, Any, Optional

from pydantic import SkipValidation, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.messages.tool import invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.pydantic import TypeBaseModel

logger = logging.getLogger(__name__)


def parse_tool_call(
    raw_tool_call: dict[str, Any],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> Optional[dict[str, Any]]:
    """Parse a single tool call.

    Args:
        raw_tool_call: The raw tool call to parse.
        partial: Whether to parse partial JSON. Default is False.
        strict: Whether to allow non-JSON-compliant strings.
            Default is False.
        return_id: Whether to return the tool call id. Default is True.

    Returns:
        The parsed tool call.

    Raises:
        OutputParserException: If the tool call is not valid JSON.
    """
    if "function" not in raw_tool_call:
        return None
    if partial:
        try:
            function_args = parse_partial_json(
                raw_tool_call["function"]["arguments"], strict=strict
            )
        except (JSONDecodeError, TypeError):  # None args raise TypeError
            return None
    else:
        try:
            function_args = json.loads(
                raw_tool_call["function"]["arguments"], strict=strict
            )
        except JSONDecodeError as e:
            msg = (
                f"Function {raw_tool_call['function']['name']} arguments:\n\n"
                f"{raw_tool_call['function']['arguments']}\n\nare not valid JSON. "
                f"Received JSONDecodeError {e}"
            )
            raise OutputParserException(msg) from e
    parsed = {
        "name": raw_tool_call["function"]["name"] or "",
        "args": function_args or {},
    }
    if return_id:
        parsed["id"] = raw_tool_call.get("id")
        parsed = create_tool_call(**parsed)  # type: ignore[assignment,arg-type]
    return parsed


def make_invalid_tool_call(
    raw_tool_call: dict[str, Any],
    error_msg: Optional[str],
) -> InvalidToolCall:
    """Create an InvalidToolCall from a raw tool call.

    Args:
        raw_tool_call: The raw tool call.
        error_msg: The error message.

    Returns:
        An InvalidToolCall instance with the error message.
    """
    return invalid_tool_call(
        name=raw_tool_call["function"]["name"],
        args=raw_tool_call["function"]["arguments"],
        id=raw_tool_call.get("id"),
        error=error_msg,
    )


def parse_tool_calls(
    raw_tool_calls: list[dict],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> list[dict[str, Any]]:
    """Parse a list of tool calls.

    Args:
        raw_tool_calls: The raw tool calls to parse.
        partial: Whether to parse partial JSON. Default is False.
        strict: Whether to allow non-JSON-compliant strings.
            Default is False.
        return_id: Whether to return the tool call id. Default is True.

    Returns:
        The parsed tool calls.

    Raises:
        OutputParserException: If any of the tool calls are not valid JSON.
    """
    final_tools: list[dict[str, Any]] = []
    exceptions = []
    for tool_call in raw_tool_calls:
        try:
            parsed = parse_tool_call(
                tool_call, partial=partial, strict=strict, return_id=return_id
            )
            if parsed:
                final_tools.append(parsed)
        except OutputParserException as e:
            exceptions.append(str(e))
            continue
    if exceptions:
        raise OutputParserException("\n\n".join(exceptions))
    return final_tools


class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse tools from OpenAI response."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    """
    return_id: bool = False
    """Whether to return the tool call id."""
    first_tool_only: bool = False
    """Whether to return only the first tool call.

    If False, the result will be a list of tool calls, or an empty list
    if no tool calls are found.

    If true, and multiple tool calls are found, only the first one will be returned,
    and the other tool calls will be ignored.
    If no tool calls are found, None will be returned.
    """

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of tool calls.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

        Returns:
            The parsed tool calls.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in tool_calls:
                if not self.return_id:
                    _ = tool_call.pop("id")
        else:
            try:
                raw_tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
            except KeyError:
                return []
            tool_calls = parse_tool_calls(
                raw_tool_calls,
                partial=partial,
                strict=self.strict,
                return_id=self.return_id,
            )
        # for backwards compatibility
        for tc in tool_calls:
            tc["type"] = tc.pop("name")

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        return tool_calls

    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a list of tool calls.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed tool calls.
        """
        raise NotImplementedError


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    key_name: str
    """The type of tools to return."""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of tool calls.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

        Returns:
            The parsed tool calls.
        """
        parsed_result = super().parse_result(result, partial=partial)

        if self.first_tool_only:
            single_result = (
                parsed_result
                if parsed_result and parsed_result["type"] == self.key_name
                else None
            )
            if self.return_id:
                return single_result
            if single_result:
                return single_result["args"]
            return None
        parsed_result = [res for res in parsed_result if res["type"] == self.key_name]
        if not self.return_id:
            parsed_result = [res["args"] for res in parsed_result]
        return parsed_result


# Common cause of ValidationError is truncated output due to max_tokens.
_MAX_TOKENS_ERROR = (
    "Output parser received a `max_tokens` stop reason. "
    "The output is likely incomplete—please increase `max_tokens` "
    "or shorten your prompt."
)


class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response."""

    tools: Annotated[list[TypeBaseModel], SkipValidation()]
    """The tools to parse."""

    # TODO: Support more granular streaming of objects. Currently only streams once all
    # Pydantic object fields are present.
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of Pydantic objects.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If True, the output will be a JSON object containing
                all the keys that have been returned so far.
                If False, the output will be the full JSON object.
                Default is False.

        Returns:
            The parsed Pydantic objects.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        json_results = super().parse_result(result, partial=partial)
        if not json_results:
            return None if self.first_tool_only else []

        json_results = [json_results] if self.first_tool_only else json_results
        name_dict = {tool.__name__: tool for tool in self.tools}
        pydantic_objects = []
        for res in json_results:
            if not isinstance(res["args"], dict):
                if partial:
                    continue
                msg = (
                    f"Tool arguments must be specified as a dict, received: "
                    f"{res['args']}"
                )
                raise ValueError(msg)
            try:
                pydantic_objects.append(name_dict[res["type"]](**res["args"]))
            except (ValidationError, ValueError):
                if partial:
                    continue
                has_max_tokens_stop_reason = any(
                    generation.message.response_metadata.get("stop_reason")
                    == "max_tokens"
                    for generation in result
                    if isinstance(generation, ChatGeneration)
                )
                if has_max_tokens_stop_reason:
                    logger.exception(_MAX_TOKENS_ERROR)
                raise
        if self.first_tool_only:
            return pydantic_objects[0] if pydantic_objects else None
        return pydantic_objects
