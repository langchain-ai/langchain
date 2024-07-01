import json
from copy import deepcopy
from functools import partial
from operator import itemgetter
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, TypeVar, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough

from langchain_community.chat_models.openai import ChatOpenAI


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


class ChatVLLMOpenAI(ChatOpenAI):
    """vLLM OpenAI-compatible API client"""

    openai_api_key: Optional[SecretStr] = Field(default="dummy", alias="api_key")
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
        return "vllm-openai"

    @staticmethod
    def _get_instructions(schema_str: str, guided_mode: str) -> str:
        """Get the instructions to insert into the input."""
        if guided_mode == "guided_json":
            return "\n" + JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif guided_mode == "guided_regex":
            return "\n" + REGEX_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif guided_mode == "guided_choice":
            return "\n" + CHOICE_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        elif guided_mode == "guided_grammar":
            return "\n" + GRAMMAR_FORMAT_INSTRUCTIONS.format(schema=schema_str)
        else:
            raise ValueError(f"Unsupported guided_mode {guided_mode}")

    @staticmethod
    def _insert_instructions(
        input: LanguageModelInput, instructions: str
    ) -> LanguageModelInput:
        """Insert instructions into the input.
        While `guided_json` ensures structured output by post-processing of the logits,
        we still need to pass the instructions to the model such that it knows what
        structure is expected for increased accuracy."""
        _input = deepcopy(input)
        try:
            if isinstance(_input, StringPromptValue):
                _input.text = _input.text + instructions
            elif isinstance(_input, ChatPromptValue):
                last_message = _input.messages[-1]
                last_message.content = last_message.content + instructions
            elif isinstance(_input, str):
                _input = _input + instructions
            elif isinstance(_input, Sequence[MessageLikeRepresentation]):
                _input[-1].content = _input[-1].content + instructions
            else:
                _input = _input + instructions
        except Exception as e:
            raise ValueError(f"Unsupported input type {type(_input)}") from e
        return _input

    def with_structured_output(
        self,
        schema: Optional[_StructuredInput] = None,
        *,
        include_raw: bool = False,
        guided_mode: Literal[
            "guided_json", "guided_regex", "guided_choice", "guided_grammar"
        ] = "guided_json",
        guided_decoding_backend: Literal["outlines", "lm-format-enforcer"] = "outlines",
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _StructuredOutput]:
        """Use guided generation to ensure structured output.

        Args:
            schema: The `schema` specifies the expected output structure.It can be a
            Pydantic class, a JSON schema, a regex, a list of choices, or a grammar.
            include_raw: Whether to include the raw output in the structured output.
            guided_mode: The `guided_mode` specifies how to guide the model. It can be
            one of `guided_json`, `guided_regex`, `guided_choice`, or `guided_grammar`.
            Defaults to `guided_json`.
            guided_decoding_backend: The `guided_decoding_backend` specifies the backend
            to use for decoding the structured output. It can be one of `outlines` or
            `lm-format-enforcer`. Defaults to `outlines`.
            instructions: The `instructions` specifies the instructions to insert into
            the input. If not provided, it is automatically generated based on the
            `schema`.

        Returns:
            A Runnable that takes any `LanguageModelInput` and returns structured output

        Example:
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

        """
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        is_pydantic_class = _is_pydantic_class(schema)

        if is_pydantic_class:
            schema_str = schema.schema_json()
        elif isinstance(schema, dict) or isinstance(schema, list):
            schema_str = json.dumps(schema)
        elif isinstance(schema, str):
            schema_str = schema
        elif schema is None:
            if guided_mode == "guided_json":
                # Use generic json schema to enforce json output
                schema = {"type": "object", "additionalProperties": True}
                schema_str = json.dumps(schema)
            else:
                raise ValueError(
                    "Only `guided_json` guided_mode is supported for schema=None. "
                    f"Got {guided_mode}."
                )
        else:
            raise ValueError(f"Unsupported schema type {type(schema)}")

        if is_pydantic_class:
            output_parser = PydanticOutputParser(pydantic_object=schema)
        else:
            if guided_mode == "guided_json":
                output_parser = JsonOutputParser()
            else:
                output_parser = StrOutputParser()

        # Ensure structured output using `guided_json`
        llm = self.bind(
            extra_body={
                **self.extra_body,
                guided_mode: schema_str if is_pydantic_class else schema,
                "guided_decoding_backend": guided_decoding_backend,
            }
        )

        # Insert instructions into the input so the model knows the expected structure
        if instructions is None:
            instructions = self._get_instructions(schema_str, guided_mode)
        insert_instructions = partial(
            self._insert_instructions, instructions=instructions
        )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=insert_instructions | llm) | parser_with_fallback
        else:
            return insert_instructions | llm | output_parser
