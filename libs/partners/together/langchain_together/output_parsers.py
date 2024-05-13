
import json
from typing import Any, List, Optional

import jsonpatch  # type: ignore[import]

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation


class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse an output as the Json object."""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.
    
    See: https://docs.python.org/3/library/json.html#encoders-and-decoders
    
    Useful when the parsed output may include unicode characters or new lines.
    """

    args_only: bool = True
    """Whether to only return the arguments to the function call."""

    @property
    def _type(self) -> str:
        return "json_functions"

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        if len(result) != 1:
            raise OutputParserException(
                f"Expected exactly one result, but got {len(result)}"
            )
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            function_call = message.additional_kwargs["tool_calls"][0]['function']
        except KeyError as exc:
            if partial:
                return None
            else:
                raise OutputParserException(f"Could not parse function call: {exc}")
        try:
            if partial:
                try:
                    if self.args_only:
                        return parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        )
                    else:
                        return {
                            **function_call,
                            "arguments": parse_partial_json(
                                function_call["arguments"], strict=self.strict
                            ),
                        }
                except json.JSONDecodeError:
                    return None
            else:
                if self.args_only:
                    try:
                        return json.loads(
                            function_call["arguments"], strict=self.strict
                        )
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise OutputParserException(
                            f"Could not parse function call data: {exc}"
                        )
                else:
                    try:
                        return {
                            **function_call,
                            "arguments": json.loads(
                                function_call["arguments"], strict=self.strict
                            ),
                        }
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise OutputParserException(
                            f"Could not parse function call data: {exc}"
                        )
        except KeyError:
            return None

    # This method would be called by the default implementation of `parse_result`
    # but we're overriding that method so it's not needed.
    def parse(self, text: str) -> Any:
        raise NotImplementedError()