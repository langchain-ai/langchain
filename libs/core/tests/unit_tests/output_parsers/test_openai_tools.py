from typing import Any, AsyncIterator, Iterator, List

from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field

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


def test_partial_json_output_parser() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputToolsParser()

    actual = list(chain.stream(None))
    expected: list = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


async def test_partial_json_output_parser_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputToolsParser()

    actual = [p async for p in chain.astream(None)]
    expected: list = [[]] + [
        [{"type": "NameCollector", "args": chunk}] for chunk in EXPECTED_STREAMED_JSON
    ]
    assert actual == expected


def test_partial_json_output_parser_return_id() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

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


def test_partial_json_output_key_parser() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = list(chain.stream(None))
    expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
    assert actual == expected


async def test_partial_json_output_parser_key_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = [p async for p in chain.astream(None)]
    expected: list = [[]] + [[chunk] for chunk in EXPECTED_STREAMED_JSON]
    assert actual == expected


def test_partial_json_output_key_parser_first_only() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )

    assert list(chain.stream(None)) == EXPECTED_STREAMED_JSON


async def test_partial_json_output_parser_key_async_first_only() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )

    assert [p async for p in chain.astream(None)] == EXPECTED_STREAMED_JSON


class Person(BaseModel):
    age: int
    hair_color: str
    job: str


class NameCollector(BaseModel):
    """record names of all people mentioned"""

    names: List[str] = Field(..., description="all names mentioned")
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
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | PydanticToolsParser(
        tools=[NameCollector], first_tool_only=True
    )

    actual = list(chain.stream(None))
    assert actual == EXPECTED_STREAMED_PYDANTIC


async def test_partial_pydantic_output_parser_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | PydanticToolsParser(
        tools=[NameCollector], first_tool_only=True
    )

    actual = [p async for p in chain.astream(None)]
    assert actual == EXPECTED_STREAMED_PYDANTIC
