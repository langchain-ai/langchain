# mypy: disable-error-code="annotation-unchecked"
import sys
import typing
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Annotated as ExtensionsAnnotated
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)
from typing import TypedDict as TypingTypedDict

import pytest
from pydantic import BaseModel as BaseModelV2Maybe  #  pydantic: ignore
from pydantic import Field as FieldV2Maybe  #  pydantic: ignore
from typing_extensions import (
    TypedDict as ExtensionsTypedDict,
)

try:
    from typing import Annotated as TypingAnnotated  # type: ignore[attr-defined]
except ImportError:
    TypingAnnotated = ExtensionsAnnotated

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.utils.function_calling import (
    _convert_typed_dict_to_openai_function,
    convert_to_openai_function,
    tool_example_to_messages,
)


@pytest.fixture()
def pydantic() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """dummy function"""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def annotated_function() -> Callable:
    def dummy_function(
        arg1: ExtensionsAnnotated[int, "foo"],
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], "one of 'bar', 'baz'"],
    ) -> None:
        """dummy function"""

    return dummy_function


@pytest.fixture()
def function() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

    return dummy_function


@pytest.fixture()
def function_docstring_annotations() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1 (int): foo
            arg2: one of 'bar', 'baz'
        """

    return dummy_function


@pytest.fixture()
def runnable() -> Runnable:
    class Args(ExtensionsTypedDict):
        arg1: ExtensionsAnnotated[int, "foo"]
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], "one of 'bar', 'baz'"]

    def dummy_function(input_dict: Args) -> None:
        pass

    return RunnableLambda(dummy_function)


@pytest.fixture()
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):
        args_schema: type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "dummy function"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture()
def dummy_structured_tool() -> StructuredTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return StructuredTool.from_function(
        lambda x: None,
        name="dummy_function",
        description="dummy function",
        args_schema=Schema,
    )


@pytest.fixture()
def dummy_pydantic() -> type[BaseModel]:
    class dummy_function(BaseModel):  # noqa: N801
        """dummy function"""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def dummy_pydantic_v2() -> type[BaseModelV2Maybe]:
    class dummy_function(BaseModelV2Maybe):  # noqa: N801
        """dummy function"""

        arg1: int = FieldV2Maybe(..., description="foo")
        arg2: Literal["bar", "baz"] = FieldV2Maybe(
            ..., description="one of 'bar', 'baz'"
        )

    return dummy_function


@pytest.fixture()
def dummy_typing_typed_dict() -> type:
    class dummy_function(TypingTypedDict):  # noqa: N801
        """dummy function"""

        arg1: TypingAnnotated[int, ..., "foo"]  # noqa: F821
        arg2: TypingAnnotated[Literal["bar", "baz"], ..., "one of 'bar', 'baz'"]  # noqa: F722

    return dummy_function


@pytest.fixture()
def dummy_typing_typed_dict_docstring() -> type:
    class dummy_function(TypingTypedDict):  # noqa: N801
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

        arg1: int
        arg2: Literal["bar", "baz"]

    return dummy_function


@pytest.fixture()
def dummy_extensions_typed_dict() -> type:
    class dummy_function(ExtensionsTypedDict):  # noqa: N801
        """dummy function"""

        arg1: ExtensionsAnnotated[int, ..., "foo"]
        arg2: ExtensionsAnnotated[Literal["bar", "baz"], ..., "one of 'bar', 'baz'"]

    return dummy_function


@pytest.fixture()
def dummy_extensions_typed_dict_docstring() -> type:
    class dummy_function(ExtensionsTypedDict):  # noqa: N801
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """

        arg1: int
        arg2: Literal["bar", "baz"]

    return dummy_function


@pytest.fixture()
def json_schema() -> dict:
    return {
        "title": "dummy_function",
        "description": "dummy function",
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    }


@pytest.fixture()
def anthropic_tool() -> dict:
    return {
        "name": "dummy_function",
        "description": "dummy function",
        "input_schema": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }


@pytest.fixture()
def bedrock_converse_tool() -> dict:
    return {
        "toolSpec": {
            "name": "dummy_function",
            "description": "dummy function",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "arg1": {"description": "foo", "type": "integer"},
                        "arg2": {
                            "description": "one of 'bar', 'baz'",
                            "enum": ["bar", "baz"],
                            "type": "string",
                        },
                    },
                    "required": ["arg1", "arg2"],
                }
            },
        }
    }


class Dummy:
    def dummy_function(self, arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


class DummyWithClassMethod:
    @classmethod
    def dummy_function(cls, arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """


def test_convert_to_openai_function(
    pydantic: type[BaseModel],
    function: Callable,
    function_docstring_annotations: Callable,
    dummy_structured_tool: StructuredTool,
    dummy_tool: BaseTool,
    json_schema: dict,
    anthropic_tool: dict,
    bedrock_converse_tool: dict,
    annotated_function: Callable,
    dummy_pydantic: type[BaseModel],
    runnable: Runnable,
    dummy_typing_typed_dict: type,
    dummy_typing_typed_dict_docstring: type,
    dummy_extensions_typed_dict: type,
    dummy_extensions_typed_dict_docstring: type,
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }

    for fn in (
        pydantic,
        function,
        function_docstring_annotations,
        dummy_structured_tool,
        dummy_tool,
        json_schema,
        anthropic_tool,
        bedrock_converse_tool,
        expected,
        Dummy.dummy_function,
        DummyWithClassMethod.dummy_function,
        annotated_function,
        dummy_pydantic,
        dummy_typing_typed_dict,
        dummy_typing_typed_dict_docstring,
        dummy_extensions_typed_dict,
        dummy_extensions_typed_dict_docstring,
    ):
        actual = convert_to_openai_function(fn)  # type: ignore
        assert actual == expected

    # Test runnables
    actual = convert_to_openai_function(runnable.as_tool(description="dummy function"))
    parameters = {
        "type": "object",
        "properties": {
            "arg1": {"type": "integer"},
            "arg2": {
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    }
    runnable_expected = expected.copy()
    runnable_expected["parameters"] = parameters
    assert actual == runnable_expected

    # Test simple Tool
    def my_function(input_string: str) -> str:
        pass

    tool = Tool(
        name="dummy_function",
        func=my_function,
        description="test description",
    )
    actual = convert_to_openai_function(tool)
    expected = {
        "name": "dummy_function",
        "description": "test description",
        "parameters": {
            "properties": {"__arg1": {"title": "__arg1", "type": "string"}},
            "required": ["__arg1"],
            "type": "object",
        },
    }
    assert actual == expected


@pytest.mark.xfail(reason="Direct pydantic v2 models not yet supported")
def test_convert_to_openai_function_nested_v2() -> None:
    class NestedV2(BaseModelV2Maybe):
        nested_v2_arg1: int = FieldV2Maybe(..., description="foo")
        nested_v2_arg2: Literal["bar", "baz"] = FieldV2Maybe(
            ..., description="one of 'bar', 'baz'"
        )

    def my_function(arg1: NestedV2) -> None:
        """dummy function"""

    convert_to_openai_function(my_function)


def test_convert_to_openai_function_nested() -> None:
    class Nested(BaseModel):
        nested_arg1: int = Field(..., description="foo")
        nested_arg2: Literal["bar", "baz"] = Field(
            ..., description="one of 'bar', 'baz'"
        )

    def my_function(arg1: Nested) -> None:
        """dummy function"""

    expected = {
        "name": "my_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "object",
                    "properties": {
                        "nested_arg1": {"type": "integer", "description": "foo"},
                        "nested_arg2": {
                            "type": "string",
                            "enum": ["bar", "baz"],
                            "description": "one of 'bar', 'baz'",
                        },
                    },
                    "required": ["nested_arg1", "nested_arg2"],
                },
            },
            "required": ["arg1"],
        },
    }

    actual = convert_to_openai_function(my_function)
    assert actual == expected


def test_convert_to_openai_function_nested_strict() -> None:
    class Nested(BaseModel):
        nested_arg1: int = Field(..., description="foo")
        nested_arg2: Literal["bar", "baz"] = Field(
            ..., description="one of 'bar', 'baz'"
        )

    def my_function(arg1: Nested) -> None:
        """dummy function"""

    expected = {
        "name": "my_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "object",
                    "properties": {
                        "nested_arg1": {"type": "integer", "description": "foo"},
                        "nested_arg2": {
                            "type": "string",
                            "enum": ["bar", "baz"],
                            "description": "one of 'bar', 'baz'",
                        },
                    },
                    "required": ["nested_arg1", "nested_arg2"],
                    "additionalProperties": False,
                },
            },
            "required": ["arg1"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    actual = convert_to_openai_function(my_function, strict=True)
    assert actual == expected


json_schema_no_description_no_params = {
    "title": "dummy_function",
}


json_schema_no_description = {
    "title": "dummy_function",
    "type": "object",
    "properties": {
        "arg1": {"description": "foo", "type": "integer"},
        "arg2": {
            "description": "one of 'bar', 'baz'",
            "enum": ["bar", "baz"],
            "type": "string",
        },
    },
    "required": ["arg1", "arg2"],
}


anthropic_tool_no_description = {
    "name": "dummy_function",
    "input_schema": {
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    },
}


bedrock_converse_tool_no_description = {
    "toolSpec": {
        "name": "dummy_function",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "arg1": {"description": "foo", "type": "integer"},
                    "arg2": {
                        "description": "one of 'bar', 'baz'",
                        "enum": ["bar", "baz"],
                        "type": "string",
                    },
                },
                "required": ["arg1", "arg2"],
            }
        },
    }
}


openai_function_no_description = {
    "name": "dummy_function",
    "parameters": {
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    },
}


openai_function_no_description_no_params = {
    "name": "dummy_function",
}


@pytest.mark.parametrize(
    "func",
    [
        anthropic_tool_no_description,
        json_schema_no_description,
        bedrock_converse_tool_no_description,
        openai_function_no_description,
    ],
)
def test_convert_to_openai_function_no_description(func: dict) -> None:
    expected = {
        "name": "dummy_function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }
    actual = convert_to_openai_function(func)
    assert actual == expected


@pytest.mark.parametrize(
    "func",
    [
        json_schema_no_description_no_params,
        openai_function_no_description_no_params,
    ],
)
def test_convert_to_openai_function_no_description_no_params(func: dict) -> None:
    expected = {
        "name": "dummy_function",
    }
    actual = convert_to_openai_function(func)
    assert actual == expected


@pytest.mark.xfail(
    reason="Pydantic converts Optional[str] to str in .model_json_schema()"
)
def test_function_optional_param() -> None:
    @tool
    def func5(
        a: Optional[str],
        b: str,
        c: Optional[list[Optional[str]]],
    ) -> None:
        """A test function"""

    func = convert_to_openai_function(func5)
    req = func["parameters"]["required"]
    assert set(req) == {"b"}


def test_function_no_params() -> None:
    def nullary_function() -> None:
        """nullary function"""

    func = convert_to_openai_function(nullary_function)
    req = func["parameters"].get("required")
    assert not req


class FakeCall(BaseModel):
    data: str


def test_valid_example_conversion() -> None:
    expected_messages = [
        HumanMessage(content="This is a valid example"),
        AIMessage(content="", additional_kwargs={"tool_calls": []}),
    ]
    assert (
        tool_example_to_messages(input="This is a valid example", tool_calls=[])
        == expected_messages
    )


def test_multiple_tool_calls() -> None:
    messages = tool_example_to_messages(
        input="This is an example",
        tool_calls=[
            FakeCall(data="ToolCall1"),
            FakeCall(data="ToolCall2"),
            FakeCall(data="ToolCall3"),
        ],
    )
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], ToolMessage)
    assert isinstance(messages[4], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data":"ToolCall1"}'},
        },
        {
            "id": messages[3].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data":"ToolCall2"}'},
        },
        {
            "id": messages[4].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data":"ToolCall3"}'},
        },
    ]


def test_tool_outputs() -> None:
    messages = tool_example_to_messages(
        input="This is an example",
        tool_calls=[
            FakeCall(data="ToolCall1"),
        ],
        tool_outputs=["Output1"],
    )
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data":"ToolCall1"}'},
        },
    ]
    assert messages[2].content == "Output1"

    # Test final AI response
    messages = tool_example_to_messages(
        input="This is an example",
        tool_calls=[
            FakeCall(data="ToolCall1"),
        ],
        tool_outputs=["Output1"],
        ai_response="The output is Output1",
    )
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage)
    response = messages[3]
    assert response.content == "The output is Output1"
    assert not response.tool_calls


@pytest.mark.parametrize("use_extension_typed_dict", [True, False])
@pytest.mark.parametrize("use_extension_annotated", [True, False])
def test__convert_typed_dict_to_openai_function(
    use_extension_typed_dict: bool, use_extension_annotated: bool
) -> None:
    typed_dict = ExtensionsTypedDict if use_extension_typed_dict else TypingTypedDict
    annotated = TypingAnnotated if use_extension_annotated else TypingAnnotated

    class SubTool(typed_dict):
        """Subtool docstring"""

        args: annotated[dict[str, Any], {}, "this does bar"]  # noqa: F722  # type: ignore

    class Tool(typed_dict):
        """Docstring

        Args:
            arg1: foo
        """

        arg1: str
        arg2: Union[int, str, bool]
        arg3: Optional[list[SubTool]]
        arg4: annotated[Literal["bar", "baz"], ..., "this does foo"]  # noqa: F722
        arg5: annotated[Optional[float], None]
        arg6: annotated[
            Optional[Sequence[Mapping[str, tuple[Iterable[Any], SubTool]]]], []
        ]
        arg7: annotated[list[SubTool], ...]
        arg8: annotated[tuple[SubTool], ...]
        arg9: annotated[Sequence[SubTool], ...]
        arg10: annotated[Iterable[SubTool], ...]
        arg11: annotated[set[SubTool], ...]
        arg12: annotated[dict[str, SubTool], ...]
        arg13: annotated[Mapping[str, SubTool], ...]
        arg14: annotated[MutableMapping[str, SubTool], ...]
        arg15: annotated[bool, False, "flag"]  # noqa: F821  # type: ignore

    expected = {
        "name": "Tool",
        "description": "Docstring",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "string"},
                "arg2": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "string"},
                        {"type": "boolean"},
                    ]
                },
                "arg3": {
                    "type": "array",
                    "items": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg4": {
                    "description": "this does foo",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
                "arg5": {"type": "number"},
                "arg6": {
                    "default": [],
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": [
                                {"type": "array", "items": {}},
                                {
                                    "title": "SubTool",
                                    "description": "Subtool docstring",
                                    "type": "object",
                                    "properties": {
                                        "args": {
                                            "title": "Args",
                                            "description": "this does bar",
                                            "default": {},
                                            "type": "object",
                                        }
                                    },
                                },
                            ],
                        },
                    },
                },
                "arg7": {
                    "type": "array",
                    "items": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg8": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": [
                        {
                            "title": "SubTool",
                            "description": "Subtool docstring",
                            "type": "object",
                            "properties": {
                                "args": {
                                    "title": "Args",
                                    "description": "this does bar",
                                    "default": {},
                                    "type": "object",
                                }
                            },
                        }
                    ],
                },
                "arg9": {
                    "type": "array",
                    "items": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg10": {
                    "type": "array",
                    "items": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg11": {
                    "type": "array",
                    "items": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                    "uniqueItems": True,
                },
                "arg12": {
                    "type": "object",
                    "additionalProperties": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg13": {
                    "type": "object",
                    "additionalProperties": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg14": {
                    "type": "object",
                    "additionalProperties": {
                        "description": "Subtool docstring",
                        "type": "object",
                        "properties": {
                            "args": {
                                "description": "this does bar",
                                "default": {},
                                "type": "object",
                            }
                        },
                    },
                },
                "arg15": {"description": "flag", "default": False, "type": "boolean"},
            },
            "required": [
                "arg1",
                "arg2",
                "arg3",
                "arg4",
                "arg7",
                "arg8",
                "arg9",
                "arg10",
                "arg11",
                "arg12",
                "arg13",
                "arg14",
            ],
        },
    }
    actual = _convert_typed_dict_to_openai_function(Tool)
    assert actual == expected


@pytest.mark.parametrize("typed_dict", [ExtensionsTypedDict, TypingTypedDict])
def test__convert_typed_dict_to_openai_function_fail(typed_dict: type) -> None:
    class Tool(typed_dict):
        arg1: typing.MutableSet  # Pydantic 2 supports this, but pydantic v1 does not.

    # Error should be raised since we're using v1 code path here
    with pytest.raises(TypeError):
        _convert_typed_dict_to_openai_function(Tool)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires python version >= 3.10 to run."
)
def test_convert_union_type_py_39() -> None:
    @tool
    def magic_function(input: int | float) -> str:
        """Compute a magic function."""

    result = convert_to_openai_function(magic_function)
    assert result["parameters"]["properties"]["input"] == {
        "anyOf": [{"type": "integer"}, {"type": "number"}]
    }


def test_convert_to_openai_function_no_args() -> None:
    @tool
    def empty_tool() -> str:
        """No args"""
        return "foo"

    actual = convert_to_openai_function(empty_tool, strict=True)
    assert actual == {
        "name": "empty_tool",
        "description": "No args",
        "parameters": {
            "properties": {},
            "additionalProperties": False,
            "type": "object",
        },
        "strict": True,
    }
