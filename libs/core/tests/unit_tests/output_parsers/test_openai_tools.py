import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pydantic
import pytest
from pydantic import BaseModel, Field, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolCallChunk,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration

STREAMED_MESSAGES = [
    AIMessageChunk(content=""),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
                    "function": {"arguments": "", "name": "NameCollector"},
                    "type": "function",
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": '{"na', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'mes":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' ["suz', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'y", ', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": '"jerm', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'aine",', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "al', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'ex"],', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "pers', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'on":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' {"ag', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'e": 39', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ', "h', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": "air_c", "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'olor":', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "br', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'own",', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ' "job"', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": ': "c', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": "oncie", "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": None,
                    "function": {"arguments": 'rge"}}', "name": None},
                    "type": None,
                }
            ]
        },
    ),
    AIMessageChunk(content=""),
]


STREAMED_MESSAGES_WITH_TOOL_CALLS = []
for message in STREAMED_MESSAGES:
    if message.additional_kwargs:
        STREAMED_MESSAGES_WITH_TOOL_CALLS.append(
            AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                tool_call_chunks=[
                    ToolCallChunk(
                        name=chunk["function"].get("name"),
                        args=chunk["function"].get("arguments"),
                        id=chunk.get("id"),
                        index=chunk["index"],
                    )
                    for chunk in message.additional_kwargs["tool_calls"]
                ],
            )
        )
    else:
        STREAMED_MESSAGES_WITH_TOOL_CALLS.append(message)


EXPECTED_STREAMED_JSON: list[dict[str, Any]] = [
    {},
    {"names": ["suz"]},
    {"names": ["suzy"]},
    {"names": ["suzy", "jerm"]},
    {"names": ["suzy", "jermaine"]},
    {"names": ["suzy", "jermaine", "al"]},
    {"names": ["suzy", "jermaine", "alex"]},
    {"names": ["suzy", "jermaine", "alex"], "person": {}},
    {"names": ["suzy", "jermaine", "alex"], "person": {"age": 39}},
    {"names": ["suzy", "jermaine", "alex"], "person": {"age": 39, "hair_color": "br"}},
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "c"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "concie"},
    },
    {
        "names": ["suzy", "jermaine", "alex"],
        "person": {"age": 39, "hair_color": "brown", "job": "concierge"},
    },
]


def _get_iter(*, use_tool_calls: bool = False) -> Any:
    if use_tool_calls:
        list_to_iter = STREAMED_MESSAGES_WITH_TOOL_CALLS
    else:
        list_to_iter = STREAMED_MESSAGES

    def input_iter(_: Any) -> Iterator[BaseMessage]:
        yield from list_to_iter

    return input_iter


def _get_aiter(*, use_tool_calls: bool = False) -> Any:
    if use_tool_calls:
        list_to_iter = STREAMED_MESSAGES_WITH_TOOL_CALLS
    else:
        list_to_iter = STREAMED_MESSAGES

    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for msg in list_to_iter:
            yield msg

    return input_iter


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_partial_json_output_parser(*, use_tool_calls: bool) -> None:
    input_iter = _get_iter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputToolsParser()

    actual = list(chain.stream(None))
    expected: list[list[dict[str, Any]]] = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
async def test_partial_json_output_parser_async(*, use_tool_calls: bool) -> None:
    input_iter = _get_aiter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputToolsParser()

    actual = [p async for p in chain.astream(None)]
    expected: list[list[dict[str, Any]]] = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_partial_json_output_parser_return_id(*, use_tool_calls: bool) -> None:
    input_iter = _get_iter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputToolsParser(return_id=True)

    actual = list(chain.stream(None))
    expected: list[list[dict[str, Any]]] = [[]] + [
        [
            {
                "type": "NameCollector",
                "args": chunk,
                "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            }
        ]
        for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_partial_json_output_key_parser(*, use_tool_calls: bool) -> None:
    input_iter = _get_iter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = list(chain.stream(None))
    expected: list[list[dict[str, Any]]] = [[]] + [
        [chunk] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
async def test_partial_json_output_parser_key_async(*, use_tool_calls: bool) -> None:
    input_iter = _get_aiter(use_tool_calls=use_tool_calls)

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = [p async for p in chain.astream(None)]
    expected: list[list[dict[str, Any]]] = [[]] + [
        [chunk] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_partial_json_output_key_parser_first_only(*, use_tool_calls: bool) -> None:
    input_iter = _get_iter(use_tool_calls=use_tool_calls)

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON


@pytest.mark.parametrize("use_tool_calls", [False, True])
async def test_partial_json_output_parser_key_async_first_only(
    *,
    use_tool_calls: bool,
) -> None:
    input_iter = _get_aiter(use_tool_calls=use_tool_calls)

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_json_output_key_tools_parser_multiple_tools_first_only(
    *, use_tool_calls: bool
) -> None:
    # Test case from the original bug report
    def create_message() -> AIMessage:
        tool_calls_data = [
            {
                "id": "call_other",
                "function": {"name": "other", "arguments": '{"b":2}'},
                "type": "function",
            },
            {
                "id": "call_func",
                "function": {"name": "func", "arguments": '{"a":1}'},
                "type": "function",
            },
        ]

        if use_tool_calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_other", "name": "other", "args": {"b": 2}},
                    {"id": "call_func", "name": "func", "args": {"a": 1}},
                ],
            )
        return AIMessage(
            content="",
            additional_kwargs={"tool_calls": tool_calls_data},
        )

    result = [ChatGeneration(message=create_message())]

    # Test with return_id=True
    parser = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=True
    )
    output = parser.parse_result(result)  # type: ignore[arg-type]

    # Should return the func tool call, not None
    assert output is not None
    assert output["type"] == "func"
    assert output["args"] == {"a": 1}
    assert "id" in output

    # Test with return_id=False
    parser_no_id = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=False
    )
    output_no_id = parser_no_id.parse_result(result)  # type: ignore[arg-type]

    # Should return just the args
    assert output_no_id == {"a": 1}


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_json_output_key_tools_parser_multiple_tools_no_match(
    *, use_tool_calls: bool
) -> None:
    def create_message() -> AIMessage:
        tool_calls_data = [
            {
                "id": "call_other",
                "function": {"name": "other", "arguments": '{"b":2}'},
                "type": "function",
            },
            {
                "id": "call_another",
                "function": {"name": "another", "arguments": '{"c":3}'},
                "type": "function",
            },
        ]

        if use_tool_calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_other", "name": "other", "args": {"b": 2}},
                    {"id": "call_another", "name": "another", "args": {"c": 3}},
                ],
            )
        return AIMessage(
            content="",
            additional_kwargs={"tool_calls": tool_calls_data},
        )

    result = [ChatGeneration(message=create_message())]

    # Test with return_id=True, first_tool_only=True
    parser = JsonOutputKeyToolsParser(
        key_name="nonexistent", first_tool_only=True, return_id=True
    )
    output = parser.parse_result(result)  # type: ignore[arg-type]

    # Should return None when no matches
    assert output is None

    # Test with return_id=False, first_tool_only=True
    parser_no_id = JsonOutputKeyToolsParser(
        key_name="nonexistent", first_tool_only=True, return_id=False
    )
    output_no_id = parser_no_id.parse_result(result)  # type: ignore[arg-type]

    # Should return None when no matches
    assert output_no_id is None


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_json_output_key_tools_parser_multiple_matching_tools(
    *, use_tool_calls: bool
) -> None:
    def create_message() -> AIMessage:
        tool_calls_data = [
            {
                "id": "call_func1",
                "function": {"name": "func", "arguments": '{"a":1}'},
                "type": "function",
            },
            {
                "id": "call_other",
                "function": {"name": "other", "arguments": '{"b":2}'},
                "type": "function",
            },
            {
                "id": "call_func2",
                "function": {"name": "func", "arguments": '{"a":3}'},
                "type": "function",
            },
        ]

        if use_tool_calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_func1", "name": "func", "args": {"a": 1}},
                    {"id": "call_other", "name": "other", "args": {"b": 2}},
                    {"id": "call_func2", "name": "func", "args": {"a": 3}},
                ],
            )
        return AIMessage(
            content="",
            additional_kwargs={"tool_calls": tool_calls_data},
        )

    result = [ChatGeneration(message=create_message())]

    # Test with first_tool_only=True - should return first matching
    parser = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=True
    )
    output = parser.parse_result(result)  # type: ignore[arg-type]

    assert output is not None
    assert output["type"] == "func"
    assert output["args"] == {"a": 1}  # First matching tool call

    # Test with first_tool_only=False - should return all matching
    parser_all = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=False, return_id=True
    )
    output_all = parser_all.parse_result(result)  # type: ignore[arg-type]

    assert len(output_all) == 2
    assert output_all[0]["args"] == {"a": 1}
    assert output_all[1]["args"] == {"a": 3}


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_json_output_key_tools_parser_empty_results(*, use_tool_calls: bool) -> None:
    def create_message() -> AIMessage:
        if use_tool_calls:
            return AIMessage(content="", tool_calls=[])
        return AIMessage(content="", additional_kwargs={"tool_calls": []})

    result = [ChatGeneration(message=create_message())]

    # Test with first_tool_only=True
    parser = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=True
    )
    output = parser.parse_result(result)  # type: ignore[arg-type]

    # Should return None for empty results
    assert output is None

    # Test with first_tool_only=False
    parser_all = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=False, return_id=True
    )
    output_all = parser_all.parse_result(result)  # type: ignore[arg-type]

    # Should return empty list for empty results
    assert output_all == []


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_json_output_key_tools_parser_parameter_combinations(
    *, use_tool_calls: bool
) -> None:
    """Test all parameter combinations of JsonOutputKeyToolsParser."""

    def create_message() -> AIMessage:
        tool_calls_data = [
            {
                "id": "call_other",
                "function": {"name": "other", "arguments": '{"b":2}'},
                "type": "function",
            },
            {
                "id": "call_func1",
                "function": {"name": "func", "arguments": '{"a":1}'},
                "type": "function",
            },
            {
                "id": "call_func2",
                "function": {"name": "func", "arguments": '{"a":3}'},
                "type": "function",
            },
        ]

        if use_tool_calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_other", "name": "other", "args": {"b": 2}},
                    {"id": "call_func1", "name": "func", "args": {"a": 1}},
                    {"id": "call_func2", "name": "func", "args": {"a": 3}},
                ],
            )
        return AIMessage(
            content="",
            additional_kwargs={"tool_calls": tool_calls_data},
        )

    result: list[ChatGeneration] = [ChatGeneration(message=create_message())]

    # Test: first_tool_only=True, return_id=True
    parser1 = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=True
    )
    output1 = parser1.parse_result(result)  # type: ignore[arg-type]
    assert output1["type"] == "func"
    assert output1["args"] == {"a": 1}
    assert "id" in output1

    # Test: first_tool_only=True, return_id=False
    parser2 = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=True, return_id=False
    )
    output2 = parser2.parse_result(result)  # type: ignore[arg-type]
    assert output2 == {"a": 1}

    # Test: first_tool_only=False, return_id=True
    parser3 = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=False, return_id=True
    )
    output3 = parser3.parse_result(result)  # type: ignore[arg-type]
    assert len(output3) == 2
    assert all("id" in item for item in output3)
    assert output3[0]["args"] == {"a": 1}
    assert output3[1]["args"] == {"a": 3}

    # Test: first_tool_only=False, return_id=False
    parser4 = JsonOutputKeyToolsParser(
        key_name="func", first_tool_only=False, return_id=False
    )
    output4 = parser4.parse_result(result)  # type: ignore[arg-type]
    assert output4 == [{"a": 1}, {"a": 3}]


class Person(BaseModel):
    age: int
    hair_color: str
    job: str


class NameCollector(BaseModel):
    """record names of all people mentioned."""

    names: list[str] = Field(..., description="all names mentioned")
    person: Person = Field(..., description="info about the main subject")


# Expected to change when we support more granular pydantic streaming.
EXPECTED_STREAMED_PYDANTIC = [
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="c"),
    ),
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="concie"),
    ),
    NameCollector(
        names=["suzy", "jermaine", "alex"],
        person=Person(age=39, hair_color="brown", job="concierge"),
    ),
]


def test_partial_pydantic_output_parser() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_iter(use_tool_calls=use_tool_calls)

        chain = input_iter | PydanticToolsParser(
            tools=[NameCollector], first_tool_only=True
        )

        actual = list(chain.stream(None))
        assert actual == EXPECTED_STREAMED_PYDANTIC


async def test_partial_pydantic_output_parser_async() -> None:
    for use_tool_calls in [False, True]:
        input_iter = _get_aiter(use_tool_calls=use_tool_calls)

        chain = input_iter | PydanticToolsParser(
            tools=[NameCollector], first_tool_only=True
        )

        actual = [p async for p in chain.astream(None)]
        assert actual == EXPECTED_STREAMED_PYDANTIC


def test_parse_with_different_pydantic_2_v1() -> None:
    """Test with pydantic.v1.BaseModel from pydantic 2."""

    class Forecast(pydantic.v1.BaseModel):
        temperature: int
        forecast: str

    # Can't get pydantic to work here due to the odd typing of tryig to support
    # both v1 and v2 in the same codebase.
    parser = PydanticToolsParser(tools=[Forecast])
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_OwL7f5PE",
                "name": "Forecast",
                "args": {"temperature": 20, "forecast": "Sunny"},
            }
        ],
    )

    generation = ChatGeneration(
        message=message,
    )

    assert parser.parse_result([generation]) == [
        Forecast(
            temperature=20,
            forecast="Sunny",
        )
    ]


def test_parse_with_different_pydantic_2_proper() -> None:
    """Test with pydantic.BaseModel from pydantic 2."""

    class Forecast(BaseModel):
        temperature: int
        forecast: str

    # Can't get pydantic to work here due to the odd typing of tryig to support
    # both v1 and v2 in the same codebase.
    parser = PydanticToolsParser(tools=[Forecast])
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_OwL7f5PE",
                "name": "Forecast",
                "args": {"temperature": 20, "forecast": "Sunny"},
            }
        ],
    )

    generation = ChatGeneration(
        message=message,
    )

    assert parser.parse_result([generation]) == [
        Forecast(
            temperature=20,
            forecast="Sunny",
        )
    ]


def test_max_tokens_error(caplog: Any) -> None:
    parser = PydanticToolsParser(tools=[NameCollector], first_tool_only=True)
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_OwL7f5PE",
                "name": "NameCollector",
                "args": {"names": ["suz", "jerm"]},
            }
        ],
        response_metadata={"stop_reason": "max_tokens"},
    )
    with pytest.raises(ValidationError):
        _ = parser.invoke(message)
    assert any(
        "`max_tokens` stop reason" in msg and record.levelname == "ERROR"
        for record, msg in zip(caplog.records, caplog.messages, strict=False)
    )


def test_pydantic_tools_parser_with_mixed_pydantic_versions() -> None:
    """Test PydanticToolsParser with both Pydantic v1 and v2 models."""
    # For Python 3.14+ compatibility, use create_model for Pydantic v1
    if sys.version_info >= (3, 14):
        WeatherV1 = pydantic.v1.create_model(  # noqa: N806
            "WeatherV1",
            __doc__="Weather information using Pydantic v1.",
            temperature=(int, ...),
            conditions=(str, ...),
        )
    else:

        class WeatherV1(pydantic.v1.BaseModel):
            """Weather information using Pydantic v1."""

            temperature: int
            conditions: str

    class LocationV2(BaseModel):
        """Location information using Pydantic v2."""

        city: str
        country: str

    # Test with Pydantic v1 model
    parser_v1 = PydanticToolsParser(tools=[WeatherV1])
    message_v1 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_weather",
                "name": "WeatherV1",
                "args": {"temperature": 25, "conditions": "sunny"},
            }
        ],
    )
    generation_v1 = ChatGeneration(message=message_v1)
    result_v1 = parser_v1.parse_result([generation_v1])

    assert len(result_v1) == 1
    assert isinstance(result_v1[0], WeatherV1)
    assert result_v1[0].temperature == 25  # type: ignore[attr-defined,unused-ignore]
    assert result_v1[0].conditions == "sunny"  # type: ignore[attr-defined,unused-ignore]

    # Test with Pydantic v2 model
    parser_v2 = PydanticToolsParser(tools=[LocationV2])
    message_v2 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_location",
                "name": "LocationV2",
                "args": {"city": "Paris", "country": "France"},
            }
        ],
    )
    generation_v2 = ChatGeneration(message=message_v2)
    result_v2 = parser_v2.parse_result([generation_v2])

    assert len(result_v2) == 1
    assert isinstance(result_v2[0], LocationV2)
    assert result_v2[0].city == "Paris"
    assert result_v2[0].country == "France"

    # Test with both v1 and v2 models
    parser_mixed = PydanticToolsParser(tools=[WeatherV1, LocationV2])
    message_mixed = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_weather",
                "name": "WeatherV1",
                "args": {"temperature": 20, "conditions": "cloudy"},
            },
            {
                "id": "call_location",
                "name": "LocationV2",
                "args": {"city": "London", "country": "UK"},
            },
        ],
    )
    generation_mixed = ChatGeneration(message=message_mixed)
    result_mixed = parser_mixed.parse_result([generation_mixed])

    assert len(result_mixed) == 2
    assert isinstance(result_mixed[0], WeatherV1)
    assert result_mixed[0].temperature == 20  # type: ignore[attr-defined,unused-ignore]
    assert isinstance(result_mixed[1], LocationV2)
    assert result_mixed[1].city == "London"


def test_pydantic_tools_parser_with_custom_title() -> None:
    """Test PydanticToolsParser with Pydantic v2 model using custom title."""

    class CustomTitleTool(BaseModel):
        """Tool with custom title in model config."""

        model_config = {"title": "MyCustomToolName"}

        value: int
        description: str

    # Test with custom title - tool should be callable by custom name
    parser = PydanticToolsParser(tools=[CustomTitleTool])
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_custom",
                "name": "MyCustomToolName",
                "args": {"value": 42, "description": "test"},
            }
        ],
    )
    generation = ChatGeneration(message=message)
    result = parser.parse_result([generation])

    assert len(result) == 1
    assert isinstance(result[0], CustomTitleTool)
    assert result[0].value == 42
    assert result[0].description == "test"


def test_pydantic_tools_parser_name_dict_fallback() -> None:
    """Test that name_dict properly falls back to __name__ when title is None."""

    class ToolWithoutTitle(BaseModel):
        """Tool without explicit title."""

        data: str

    # Ensure model_config doesn't have a title or it's None
    # (This is the default behavior)
    parser = PydanticToolsParser(tools=[ToolWithoutTitle])
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_no_title",
                "name": "ToolWithoutTitle",
                "args": {"data": "test_data"},
            }
        ],
    )
    generation = ChatGeneration(message=message)
    result = parser.parse_result([generation])

    assert len(result) == 1
    assert isinstance(result[0], ToolWithoutTitle)
    assert result[0].data == "test_data"


def test_pydantic_tools_parser_with_nested_models() -> None:
    """Test PydanticToolsParser with nested Pydantic v1 and v2 models."""
    # Nested v1 models
    if sys.version_info >= (3, 14):
        AddressV1 = pydantic.v1.create_model(  # noqa: N806
            "AddressV1",
            __doc__="Address using Pydantic v1.",
            street=(str, ...),
            city=(str, ...),
            zip_code=(str, ...),
        )
        PersonV1 = pydantic.v1.create_model(  # noqa: N806
            "PersonV1",
            __doc__="Person with nested address using Pydantic v1.",
            name=(str, ...),
            age=(int, ...),
            address=(AddressV1, ...),
        )
    else:

        class AddressV1(pydantic.v1.BaseModel):
            """Address using Pydantic v1."""

            street: str
            city: str
            zip_code: str

        class PersonV1(pydantic.v1.BaseModel):
            """Person with nested address using Pydantic v1."""

            name: str
            age: int
            address: AddressV1

    # Nested v2 models
    class CoordinatesV2(BaseModel):
        """Coordinates using Pydantic v2."""

        latitude: float
        longitude: float

    class LocationV2(BaseModel):
        """Location with nested coordinates using Pydantic v2."""

        name: str
        coordinates: CoordinatesV2

    # Test with nested Pydantic v1 model
    parser_v1 = PydanticToolsParser(tools=[PersonV1])
    message_v1 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_person",
                "name": "PersonV1",
                "args": {
                    "name": "Alice",
                    "age": 30,
                    "address": {
                        "street": "123 Main St",
                        "city": "Springfield",
                        "zip_code": "12345",
                    },
                },
            }
        ],
    )
    generation_v1 = ChatGeneration(message=message_v1)
    result_v1 = parser_v1.parse_result([generation_v1])

    assert len(result_v1) == 1
    assert isinstance(result_v1[0], PersonV1)
    assert result_v1[0].name == "Alice"  # type: ignore[attr-defined,unused-ignore]
    assert result_v1[0].age == 30  # type: ignore[attr-defined,unused-ignore]
    assert isinstance(result_v1[0].address, AddressV1)  # type: ignore[attr-defined,unused-ignore]
    assert result_v1[0].address.street == "123 Main St"  # type: ignore[attr-defined,unused-ignore]
    assert result_v1[0].address.city == "Springfield"  # type: ignore[attr-defined,unused-ignore]

    # Test with nested Pydantic v2 model
    parser_v2 = PydanticToolsParser(tools=[LocationV2])
    message_v2 = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_location",
                "name": "LocationV2",
                "args": {
                    "name": "Eiffel Tower",
                    "coordinates": {"latitude": 48.8584, "longitude": 2.2945},
                },
            }
        ],
    )
    generation_v2 = ChatGeneration(message=message_v2)
    result_v2 = parser_v2.parse_result([generation_v2])

    assert len(result_v2) == 1
    assert isinstance(result_v2[0], LocationV2)
    assert result_v2[0].name == "Eiffel Tower"
    assert isinstance(result_v2[0].coordinates, CoordinatesV2)
    assert result_v2[0].coordinates.latitude == 48.8584
    assert result_v2[0].coordinates.longitude == 2.2945

    # Test with both nested models in one message
    parser_mixed = PydanticToolsParser(tools=[PersonV1, LocationV2])
    message_mixed = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_person",
                "name": "PersonV1",
                "args": {
                    "name": "Bob",
                    "age": 25,
                    "address": {
                        "street": "456 Oak Ave",
                        "city": "Portland",
                        "zip_code": "97201",
                    },
                },
            },
            {
                "id": "call_location",
                "name": "LocationV2",
                "args": {
                    "name": "Golden Gate Bridge",
                    "coordinates": {"latitude": 37.8199, "longitude": -122.4783},
                },
            },
        ],
    )
    generation_mixed = ChatGeneration(message=message_mixed)
    result_mixed = parser_mixed.parse_result([generation_mixed])

    assert len(result_mixed) == 2
    assert isinstance(result_mixed[0], PersonV1)
    assert result_mixed[0].name == "Bob"  # type: ignore[attr-defined,unused-ignore]
    assert result_mixed[0].address.city == "Portland"  # type: ignore[attr-defined,unused-ignore]
    assert isinstance(result_mixed[1], LocationV2)
    assert result_mixed[1].name == "Golden Gate Bridge"
    assert result_mixed[1].coordinates.latitude == 37.8199


def test_pydantic_tools_parser_with_optional_fields() -> None:
    """Test PydanticToolsParser with optional fields in v1 and v2 models."""
    if sys.version_info >= (3, 14):
        ProductV1 = pydantic.v1.create_model(  # noqa: N806
            "ProductV1",
            __doc__="Product with optional fields using Pydantic v1.",
            name=(str, ...),
            price=(float, ...),
            description=(str | None, None),
            stock=(int, 0),
        )
    else:

        class ProductV1(pydantic.v1.BaseModel):
            """Product with optional fields using Pydantic v1."""

            name: str
            price: float
            description: str | None = None
            stock: int = 0

    # v2 model with optional fields
    class UserV2(BaseModel):
        """User with optional fields using Pydantic v2."""

        username: str
        email: str
        bio: str | None = None
        age: int | None = None

    # Test v1 with all fields provided
    parser_v1_full = PydanticToolsParser(tools=[ProductV1])
    message_v1_full = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_product_full",
                "name": "ProductV1",
                "args": {
                    "name": "Laptop",
                    "price": 999.99,
                    "description": "High-end laptop",
                    "stock": 50,
                },
            }
        ],
    )
    generation_v1_full = ChatGeneration(message=message_v1_full)
    result_v1_full = parser_v1_full.parse_result([generation_v1_full])

    assert len(result_v1_full) == 1
    assert isinstance(result_v1_full[0], ProductV1)
    assert result_v1_full[0].name == "Laptop"  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_full[0].price == 999.99  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_full[0].description == "High-end laptop"  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_full[0].stock == 50  # type: ignore[attr-defined,unused-ignore]

    # Test v1 with only required fields
    parser_v1_minimal = PydanticToolsParser(tools=[ProductV1])
    message_v1_minimal = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_product_minimal",
                "name": "ProductV1",
                "args": {"name": "Mouse", "price": 29.99},
            }
        ],
    )
    generation_v1_minimal = ChatGeneration(message=message_v1_minimal)
    result_v1_minimal = parser_v1_minimal.parse_result([generation_v1_minimal])

    assert len(result_v1_minimal) == 1
    assert isinstance(result_v1_minimal[0], ProductV1)
    assert result_v1_minimal[0].name == "Mouse"  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_minimal[0].price == 29.99  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_minimal[0].description is None  # type: ignore[attr-defined,unused-ignore]
    assert result_v1_minimal[0].stock == 0  # type: ignore[attr-defined,unused-ignore]

    # Test v2 with all fields provided
    parser_v2_full = PydanticToolsParser(tools=[UserV2])
    message_v2_full = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_user_full",
                "name": "UserV2",
                "args": {
                    "username": "john_doe",
                    "email": "john@example.com",
                    "bio": "Software developer",
                    "age": 28,
                },
            }
        ],
    )
    generation_v2_full = ChatGeneration(message=message_v2_full)
    result_v2_full = parser_v2_full.parse_result([generation_v2_full])

    assert len(result_v2_full) == 1
    assert isinstance(result_v2_full[0], UserV2)
    assert result_v2_full[0].username == "john_doe"
    assert result_v2_full[0].email == "john@example.com"
    assert result_v2_full[0].bio == "Software developer"
    assert result_v2_full[0].age == 28

    # Test v2 with only required fields
    parser_v2_minimal = PydanticToolsParser(tools=[UserV2])
    message_v2_minimal = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_user_minimal",
                "name": "UserV2",
                "args": {"username": "jane_smith", "email": "jane@example.com"},
            }
        ],
    )
    generation_v2_minimal = ChatGeneration(message=message_v2_minimal)
    result_v2_minimal = parser_v2_minimal.parse_result([generation_v2_minimal])

    assert len(result_v2_minimal) == 1
    assert isinstance(result_v2_minimal[0], UserV2)
    assert result_v2_minimal[0].username == "jane_smith"
    assert result_v2_minimal[0].email == "jane@example.com"
    assert result_v2_minimal[0].bio is None
    assert result_v2_minimal[0].age is None

    # Test mixed v1 and v2 with partial optional fields
    parser_mixed = PydanticToolsParser(tools=[ProductV1, UserV2])
    message_mixed = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_product",
                "name": "ProductV1",
                "args": {"name": "Keyboard", "price": 79.99, "stock": 100},
            },
            {
                "id": "call_user",
                "name": "UserV2",
                "args": {
                    "username": "alice",
                    "email": "alice@example.com",
                    "age": 35,
                },
            },
        ],
    )
    generation_mixed = ChatGeneration(message=message_mixed)
    result_mixed = parser_mixed.parse_result([generation_mixed])

    assert len(result_mixed) == 2
    assert isinstance(result_mixed[0], ProductV1)
    assert result_mixed[0].name == "Keyboard"  # type: ignore[attr-defined,unused-ignore]
    assert result_mixed[0].description is None  # type: ignore[attr-defined,unused-ignore]
    assert result_mixed[0].stock == 100  # type: ignore[attr-defined,unused-ignore]
    assert isinstance(result_mixed[1], UserV2)
    assert result_mixed[1].username == "alice"
    assert result_mixed[1].bio is None
    assert result_mixed[1].age == 35


def test_parse_tool_call_with_none_arguments() -> None:
    """Test parse_tool_call handles None arguments for parameter-less tools.

    When an LLM calls a tool that has no parameters, some providers return
    None for the arguments field instead of an empty string or "{}".
    This should not raise an error.

    See: https://github.com/langchain-ai/langchain/issues/34123
    """
    # Test case from issue #34123: arguments is None
    raw_tool_call = {
        "function": {"arguments": None, "name": "orderStatus"},
        "id": "chatcmpl-tool-8b1f759d874b412e931e64cf6f57bdcc",
        "type": "function",
    }

    # This should not raise an error - should return parsed tool call with empty args
    result = parse_tool_call(raw_tool_call, return_id=True)

    assert result is not None
    assert result["name"] == "orderStatus"
    assert result["args"] == {}
    assert result["id"] == "chatcmpl-tool-8b1f759d874b412e931e64cf6f57bdcc"


def test_parse_tool_call_with_empty_string_arguments() -> None:
    """Test parse_tool_call handles empty string arguments."""
    raw_tool_call = {
        "function": {"arguments": "", "name": "getStatus"},
        "id": "call_123",
        "type": "function",
    }

    # Empty string should be treated as empty args
    result = parse_tool_call(raw_tool_call, return_id=True)

    assert result is not None
    assert result["name"] == "getStatus"
    assert result["args"] == {}
    assert result["id"] == "call_123"


def test_parse_tool_call_with_valid_arguments() -> None:
    """Test parse_tool_call works normally with valid JSON arguments."""
    raw_tool_call = {
        "function": {"arguments": '{"param": "value"}', "name": "myTool"},
        "id": "call_456",
        "type": "function",
    }

    result = parse_tool_call(raw_tool_call, return_id=True)

    assert result is not None
    assert result["name"] == "myTool"
    assert result["args"] == {"param": "value"}
    assert result["id"] == "call_456"


def test_parse_tool_call_partial_mode_with_none_arguments() -> None:
    """Test parse_tool_call in partial mode handles None arguments."""
    raw_tool_call = {
        "function": {"arguments": None, "name": "streamingTool"},
        "id": "call_789",
        "type": "function",
    }

    # Partial mode should return None for None arguments (existing behavior)
    result = parse_tool_call(raw_tool_call, partial=True, return_id=True)

    # In partial mode, None arguments returns None (incomplete tool call)
    assert result is None


@pytest.mark.parametrize("partial", [False, True])
def test_pydantic_tools_parser_unknown_tool_raises_output_parser_exception(
    partial: bool,  # noqa: FBT001
) -> None:
    class KnownTool(BaseModel):
        value: int

    parser = PydanticToolsParser(tools=[KnownTool])
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_unknown",
                "name": "UnknownTool",
                "args": {"value": 1},
            }
        ],
    )
    generation = ChatGeneration(message=message)

    with pytest.raises(OutputParserException) as excinfo:
        parser.parse_result([generation], partial=partial)

    msg = str(excinfo.value)
    assert "Unknown tool type" in msg
    assert "UnknownTool" in msg
