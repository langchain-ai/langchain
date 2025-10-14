from collections.abc import AsyncIterator, Iterator
from typing import Any

import pydantic
import pytest
from pydantic import BaseModel, Field, ValidationError

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
)
from langchain_core.outputs import ChatGeneration

STREAMED_MESSAGES: list = [
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


EXPECTED_STREAMED_JSON = [
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
    expected: list = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
async def test_partial_json_output_parser_async(*, use_tool_calls: bool) -> None:
    input_iter = _get_aiter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputToolsParser()

    actual = [p async for p in chain.astream(None)]
    expected: list = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
def test_partial_json_output_parser_return_id(*, use_tool_calls: bool) -> None:
    input_iter = _get_iter(use_tool_calls=use_tool_calls)
    chain = input_iter | JsonOutputToolsParser(return_id=True)

    actual = list(chain.stream(None))
    expected: list = [[]] + [
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
    expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
    assert actual == expected


@pytest.mark.parametrize("use_tool_calls", [False, True])
async def test_partial_json_output_parser_key_async(*, use_tool_calls: bool) -> None:
    input_iter = _get_aiter(use_tool_calls=use_tool_calls)

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = [p async for p in chain.astream(None)]
    expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
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
