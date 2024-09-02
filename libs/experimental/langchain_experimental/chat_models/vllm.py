import json
from copy import deepcopy
from functools import partial
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core._api.beta_decorator import beta
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, MessageLikeRepresentation
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fall back to community version if openai is not installed
    # The community version does not implement `bind_tools`.
    from langchain_community.chat_models.openai import (  # type: ignore[assignment]
        ChatOpenAI,
    )


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


JSON_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema
```
{{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
```
the object `{{"foo": ["bar", "baz"]}}` is a well-formatted instance of the schema. The object `{{"properties": {{"foo": ["bar", "baz"]}}}}` is not well-formatted.

Here is the output schema you must conform to:
```
{schema}
```"""  # noqa E501

JSON_MODE_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance."""

REGEX_FORMAT_INSTRUCTIONS = """The output should be formatted as a string that conforms to the regex below.

As an example, for the regex `[a-z]+` the string `abc` is a well-formatted instance of the regex. The string `123` is not well-formatted.

Here is the output regex you must conform to:
```
{schema}
```"""  # noqa E501

CHOICE_FORMAT_INSTRUCTIONS = """The output should be one of the following choices:
```
{schema}
```"""

GRAMMAR_FORMAT_INSTRUCTIONS = """The output should be a string that conforms to the grammar below.

As an example, for the grammar
```
?start: expression

?expression: term (("+" | "-") term)*

?term: factor (("*" | "/") factor)*

?factor: NUMBER
        | "-" factor
        | "(" expression ")"

%import common.NUMBER
```
the string `2*3` is a well-formatted instance of the grammar. The string "two times three" is not well-formatted.

Here is the output grammar you must conform to:
```
{schema}
```"""  # noqa E501


_BM = TypeVar("_BM", bound=BaseModel)
_StructuredInput = Union[Dict[str, Any], str, List[str], Type[_BM]]
_StructuredOutput = Union[Dict, str, _BM]


@beta
class ChatVLLMStructured(ChatOpenAI):
    """vLLM OpenAI-compatible API client with experimental output structured."""

    openai_api_key: Optional[SecretStr] = Field(default="dummy", alias="api_key")  # type: ignore[assignment]
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_api_base: Optional[str] = Field(
        default="http://localhost:8000/v1", alias="base_url"
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    extra_body: Optional[Dict[str, Any]] = Field(default={})
    """Extra body parameters to pass to the API request."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {**super()._default_params, "extra_body": self.extra_body}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm-openai-structured"

    @staticmethod
    def _get_instructions(schema_str: str, method: str) -> str:
        """Get the instructions to insert into the input."""
        if method in ["function_calling", "guided_json"]:
            return "\n" + JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif method == "json_mode":
            return "\n" + JSON_MODE_FORMAT_INSTRUCTIONS
        elif method == "guided_regex":
            return "\n" + REGEX_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif method == "guided_choice":
            return "\n" + CHOICE_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif method == "guided_grammar":
            return "\n" + GRAMMAR_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        else:
            raise ValueError(f"Unsupported method {method}")

    @staticmethod
    def _insert_instructions(
        input: LanguageModelInput, instructions: str
    ) -> LanguageModelInput:
        """Insert instructions into the input"""
        _input = deepcopy(input)
        try:
            if isinstance(_input, StringPromptValue):
                _input.text = _input.text + instructions
            elif isinstance(_input, ChatPromptValue):
                last_message = _input.messages[-1]
                last_message.content = last_message.content + instructions  # type: ignore[operator]
            elif isinstance(_input, str):
                _input = _input + instructions
            elif isinstance(_input, Sequence[MessageLikeRepresentation]):  # type: ignore[misc]
                _input[-1].content = _input[-1].content + instructions  # type: ignore[union-attr,operator]
            else:
                _input = _input + instructions  # type: ignore[operator]
        except Exception as e:
            raise ValueError(f"Unsupported input type {type(_input)}") from e
        return _input

    @staticmethod
    def _get_schema_str(
        schema: Optional[_StructuredInput],
        method: Literal[
            "function_calling",
            "json_mode",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
        ],
    ) -> str:
        if _is_pydantic_class(schema):
            return cast(Type[BaseModel], schema).schema_json()
        elif isinstance(schema, dict) or isinstance(schema, list):
            return json.dumps(schema)
        elif isinstance(schema, str):
            return schema
        elif schema is None:
            return ""
        else:
            raise ValueError(f"Unsupported schema type {type(schema)}")

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        # remove unsupported arguments
        for key in ["parallel_tool_calls"]:
            kwargs.pop(key, None)
        try:
            return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
        except NotImplementedError:
            raise ImportError(
                "Please install `langchain-openai` to use the `bind_tools` method."
            )

    def with_guided_output(
        self,
        schema: Optional[_StructuredInput] = None,
        *,
        include_raw: bool = False,
        method: Literal[
            "guided_json", "guided_regex", "guided_choice", "guided_grammar"
        ] = "guided_json",
    ) -> Runnable[LanguageModelInput, _StructuredOutput]:
        is_pydantic_class = _is_pydantic_class(schema)
        if is_pydantic_class:
            output_parser: OutputParserLike = PydanticOutputParser(
                pydantic_object=schema  # type: ignore[arg-type]
            )
        elif method == "guided_json":
            output_parser = JsonOutputParser()
        else:
            output_parser = StrOutputParser()

        # Ensure structured output using guidance
        llm = self.bind(
            extra_body={
                **(self.extra_body or {}),
                method: (
                    cast(Type[BaseModel], schema).schema_json()
                    if is_pydantic_class
                    else schema
                ),
            }
        )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Optional[_StructuredInput] = None,
        *,
        include_raw: bool = False,
        method: Literal[
            "function_calling",
            "json_mode",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
        ] = "function_calling",
        inject_instructions: bool = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _StructuredOutput]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The `schema` specifies the expected output structure. It can be a
                Pydantic class, a JSON schema, a regex, a list of choices, or a grammar.
                If a Pydantic class then the model output will be an object of that
                class. If a dict then the model output will be a dict. With a Pydantic
                class the returned attributes will be validated, whereas with a dict
                they will not be. If a dict, then it must be a valid JSON schema. If a
                list of choices, then the model output will be one of the choices. If a
                grammar, then the model output will be a string that conforms to the
                grammar.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".
            method: The `method` specifies how to guide the model. It can be
                one of `function_calling`, `json_mode`, `guided_json`, `guided_regex`,
                `guided_choice`, or `guided_grammar`. Defaults to `guided_json`.
            inject_instructions: Whether to inject instructions based on the schema
                and method into the prompt.

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_StructuredOutput]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _StructuredOutput is returned,
                where _StructuredOutput depends on the schema and method.:

                If method is "guided_choice", "guided_regex", or "guided_grammar"
                then _StructuredOutput is a str.

                If method is "guided_json" and schema is a Pydantic class then
                _StructuredOutput is an instance of that Pydantic class.

                If method is "guided_json" and schema is a dict or str then
                _StructuredOutput is a dict.

        Example: guided_json, Pydantic schema (method="guided_json", include_raw=False):
            .. code-block:: python

                from langchain_core.pydantic_v1 import BaseModel, Field

                class CityModel(BaseModel):
                    name: str = Field(..., description="Name of the city")
                    population: int = Field(
                        ..., description="Number of inhabitants"
                    )
                    country: str = Field(..., description="Country of the city")
                    population_category: Literal[">1M", "<1M"] = Field(
                        ..., description="Population category of the city"
                    )

                llm = VLLM(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                )
                structured_llm = llm.with_structured_output(CityModel)

                structured_llm.invoke("What is the capital of France?")
                # Out:
                # name='Paris' population=2141000 country='France' population_category='>1M'

        """  # noqa E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        if method in ["function_calling", "json_mode"]:
            # rely on ChatOpenAI implementation for these modes
            structured_llm = super().with_structured_output(  # type: ignore[call-overload]
                schema=schema, include_raw=include_raw, method=method
            )
        else:
            # For guided modes, rely on vLLM guidance modes for structured output
            structured_llm = self.with_guided_output(
                schema=schema,
                include_raw=include_raw,
                method=cast(
                    Literal[
                        "guided_json", "guided_regex", "guided_choice", "guided_grammar"
                    ],
                    method,
                ),
            )

        # Insert instructions into the input so the model knows the expected structure
        # This is not (currently) handled by vLLM internally.
        if inject_instructions:
            schema_str = self._get_schema_str(schema, method)
            instructions = self._get_instructions(schema_str, method)
            insert_instructions = partial(
                self._insert_instructions, instructions=instructions
            )

            return insert_instructions | structured_llm
        else:
            return structured_llm
