"""Parsers for OpenAI functions output."""

import copy
import json
from types import GenericAlias
from typing import Any

import jsonpatch  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
    BaseGenerationOutputParser,
)
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation


class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """Parse an output that is one of sets of values."""

    args_only: bool = True
    """Whether to only return the arguments to the function call."""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.

        Raises:
            `OutputParserException`: If the output is not valid JSON.
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc

        if self.args_only:
            return func_call["arguments"]
        return func_call


class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse an output as the JSON object."""

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

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserExcept`ion: If the output is not valid JSON.
        """
        if len(result) != 1:
            msg = f"Expected exactly one result, but got {len(result)}"
            raise OutputParserException(msg)
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            function_call = message.additional_kwargs["function_call"]
        except KeyError as exc:
            if partial:
                return None
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc
        try:
            if partial:
                try:
                    if self.args_only:
                        return parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        )
                    return {
                        **function_call,
                        "arguments": parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except json.JSONDecodeError:
                    return None
            elif self.args_only:
                try:
                    return json.loads(function_call["arguments"], strict=self.strict)
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
            else:
                try:
                    return {
                        **function_call,
                        "arguments": json.loads(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
        except KeyError:
            return None

    # This method would be called by the default implementation of `parse_result`
    # but we're overriding that method so it's not needed.
    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object.
        """
        raise NotImplementedError


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the element of the JSON object."""

    key_name: str
    """The name of the key to return."""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.
        """
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a Pydantic object.

    This parser is used to parse the output of a chat model that uses OpenAI function
    format to invoke functions.

    The parser extracts the function call invocation and matches them to the Pydantic
    schema provided.

    An exception will be raised if the function call does not match the provided schema.

    Example:
        ```python
        message = AIMessage(
            content="This is a test message",
            additional_kwargs={
                "function_call": {
                    "name": "cookie",
                    "arguments": json.dumps({"name": "value", "age": 10}),
                }
            },
        )
        chat_generation = ChatGeneration(message=message)


        class Cookie(BaseModel):
            name: str
            age: int


        class Dog(BaseModel):
            species: str


        # Full output
        parser = PydanticOutputFunctionsParser(
            pydantic_schema={"cookie": Cookie, "dog": Dog}
        )
        result = parser.parse_result([chat_generation])
        ```

    """

    pydantic_schema: type[BaseModel] | dict[str, type[BaseModel]]
    """The Pydantic schema to parse the output with.

    If multiple schemas are provided, then the function name will be used to
    determine which schema to use.
    """

    @model_validator(mode="before")
    @classmethod
    def validate_schema(cls, values: dict) -> Any:
        """Validate the Pydantic schema.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If the schema is not a Pydantic schema.
        """
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = (
                isinstance(schema, type)
                and not isinstance(schema, GenericAlias)
                and issubclass(schema, BaseModel)
            )
        elif values["args_only"] and isinstance(schema, dict):
            msg = (
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
            raise ValueError(msg)
        return values

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Raises:
            ValueError: If the Pydantic schema is not valid.

        Returns:
            The parsed JSON object.
        """
        result_ = super().parse_result(result)
        if self.args_only:
            if hasattr(self.pydantic_schema, "model_validate_json"):
                pydantic_args = self.pydantic_schema.model_validate_json(result_)
            else:
                pydantic_args = self.pydantic_schema.parse_raw(result_)  # type: ignore[attr-defined]
        else:
            fn_name = result_["name"]
            args = result_["arguments"]
            if isinstance(self.pydantic_schema, dict):
                pydantic_schema = self.pydantic_schema[fn_name]
            else:
                pydantic_schema = self.pydantic_schema
            if issubclass(pydantic_schema, BaseModel):
                pydantic_args = pydantic_schema.model_validate_json(args)
            elif issubclass(pydantic_schema, BaseModelV1):
                pydantic_args = pydantic_schema.parse_raw(args)
            else:
                msg = f"Unsupported Pydantic schema: {pydantic_schema}"
                raise ValueError(msg)
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a Pydantic object."""

    attr_name: str
    """The name of the attribute to return."""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.
        """
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
