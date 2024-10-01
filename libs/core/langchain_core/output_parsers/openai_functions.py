import copy
import json
from types import GenericAlias
from typing import Any, Optional, Union

import jsonpatch  # type: ignore[import]
from pydantic import BaseModel, model_validator

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

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects. Default is False.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            raise OutputParserException(
                f"Could not parse function call: {exc}"
            ) from exc

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

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects. Default is False.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        """

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
                raise OutputParserException(
                    f"Could not parse function call: {exc}"
                ) from exc
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
                        ) from exc
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
                        ) from exc
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
        raise NotImplementedError()


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the element of the Json object."""

    key_name: str
    """The name of the key to return."""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects. Default is False.

        Returns:
            The parsed JSON object.
        """
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a pydantic object.

    This parser is used to parse the output of a ChatModel that uses
    OpenAI function format to invoke functions.

    The parser extracts the function call invocation and matches
    them to the pydantic schema provided.

    An exception will be raised if the function call does not match
    the provided schema.

    Example:

        ... code-block:: python

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
    """

    pydantic_schema: Union[type[BaseModel], dict[str, type[BaseModel]]]
    """The pydantic schema to parse the output with.

    If multiple schemas are provided, then the function name will be used to
    determine which schema to use.
    """

    @model_validator(mode="before")
    @classmethod
    def validate_schema(cls, values: dict) -> Any:
        """Validate the pydantic schema.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If the schema is not a pydantic schema.
        """
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = (
                isinstance(schema, type)
                and not isinstance(schema, GenericAlias)
                and issubclass(schema, BaseModel)
            )
        elif values["args_only"] and isinstance(schema, dict):
            raise ValueError(
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
        return values

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects. Default is False.

        Returns:
            The parsed JSON object.
        """
        _result = super().parse_result(result)
        if self.args_only:
            if hasattr(self.pydantic_schema, "model_validate_json"):
                pydantic_args = self.pydantic_schema.model_validate_json(_result)
            else:
                pydantic_args = self.pydantic_schema.parse_raw(_result)  # type: ignore
        else:
            fn_name = _result["name"]
            _args = _result["arguments"]
            if isinstance(self.pydantic_schema, dict):
                pydantic_schema = self.pydantic_schema[fn_name]
            else:
                pydantic_schema = self.pydantic_schema
            if hasattr(pydantic_schema, "model_validate_json"):
                pydantic_args = pydantic_schema.model_validate_json(_args)  # type: ignore
            else:
                pydantic_args = pydantic_schema.parse_raw(_args)  # type: ignore
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a pydantic object."""

    attr_name: str
    """The name of the attribute to return."""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects. Default is False.

        Returns:
            The parsed JSON object.
        """
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
