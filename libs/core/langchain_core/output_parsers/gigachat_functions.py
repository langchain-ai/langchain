import copy
import json
from typing import Any, Dict, List, Type, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, root_validator


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
            raise OutputParserException(
                f"Could not parse function call: {exc}"
            ) from exc

        if self.args_only:
            return func_call["arguments"]
        return func_call


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

    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]
    """The pydantic schema to parse the output with.

    If multiple schemas are provided, then the function name will be used to
    determine which schema to use.
    """

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
            pydantic_args = self.pydantic_schema(**_result)  # type: ignore
        else:
            fn_name = _result.name
            _args = json.dumps(_result.arguments)
            pydantic_args = self.pydantic_schema[fn_name].parse_raw(_args)  # type: ignore  # noqa: E501
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a pydantic object."""

    attr_name: str
    """The name of the attribute to return."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
