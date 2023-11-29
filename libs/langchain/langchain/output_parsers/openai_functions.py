import copy
import json
from typing import Any, Dict, List, Optional, Type, Union

import jsonpatch
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
    BaseGenerationOutputParser,
)
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain.output_parsers.json import parse_partial_json


class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """Parse an output that is one of sets of values."""

    args_only: bool = True
    """Whether to only return the arguments to the function call."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            raise OutputParserException(f"Could not parse function call: {exc}")

        if self.args_only:
            return func_call["arguments"]
        return func_call


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
            function_call = message.additional_kwargs["function_call"]
        except KeyError as exc:
            if partial:
                return None
            else:
                raise OutputParserException(f"Could not parse function call: {exc}")
        try:
            if partial:
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


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the element of the Json object."""

    key_name: str
    """The name of the key to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a pydantic object."""

    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]
    """The pydantic schema to parse the output with."""

    @root_validator(pre=True)
    def validate_schema(cls, values: Dict) -> Dict:
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = isinstance(schema, type) and issubclass(
                schema, BaseModel
            )
        elif values["args_only"] and isinstance(schema, Dict):
            raise ValueError(
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
        return values

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        _result = super().parse_result(result)
        if self.args_only:
            pydantic_args = self.pydantic_schema.parse_raw(_result)  # type: ignore
        else:
            fn_name = _result["name"]
            _args = _result["arguments"]
            pydantic_args = self.pydantic_schema[fn_name].parse_raw(_args)  # type: ignore  # noqa: E501
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a pydantic object."""

    attr_name: str
    """The name of the attribute to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
