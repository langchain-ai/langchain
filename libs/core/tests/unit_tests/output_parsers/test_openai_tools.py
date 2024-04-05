from typing import Any, AsyncIterator, Iterator, List

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    ToolCallChunk,
)
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
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_YpExKmDW7ByjE2dHSxVeIWlX",
                    "function": {"arguments": "", "name": "get_word_length"},
                    "type": "function",
                }
            ]
        },
        tool_call_chunks=[
            ToolCallChunk(
                name="get_word_length",
                args="",
                id="call_YpExKmDW7ByjE2dHSxVeIWlX",
                index=1,
            )
        ],
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 1,
                    "id": None,
                    "function": {"arguments": '{"word": "chr', "name": None},
                    "type": None,
                }
            ]
        },
        tool_call_chunks=[
            ToolCallChunk(name=None, args='{"word": "chr', id=None, index=1)
        ],
    ),
    AIMessageChunk(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 1,
                    "id": None,
                    "function": {"arguments": 'ysanthemum"}', "name": None},
                    "type": None,
                }
            ]
        },
        tool_call_chunks=[
            ToolCallChunk(name=None, args='ysanthemum"}', id=None, index=1)
        ],
    ),
    AIMessageChunk(content=""),
]

EXPECTED_STREAMED_JSON = [
    [{"args": {}, "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl", "type": "NameCollector"}],
    [
        {
            "args": {"names": ["suz"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jerm"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jermaine"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jermaine", "al"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jermaine", "alex"]},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jermaine", "alex"], "person": {}},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {"names": ["suzy", "jermaine", "alex"], "person": {"age": 39}},
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "br"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown", "job": "c"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown", "job": "concie"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown", "job": "concierge"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        }
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown", "job": "concierge"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        },
        {
            "args": {"word": "chr"},
            "id": "call_YpExKmDW7ByjE2dHSxVeIWlX",
            "type": "get_word_length",
        },
    ],
    [
        {
            "args": {
                "names": ["suzy", "jermaine", "alex"],
                "person": {"age": 39, "hair_color": "brown", "job": "concierge"},
            },
            "id": "call_OwL7f5PEPJTYzw9sQlNJtCZl",
            "type": "NameCollector",
        },
        {
            "args": {"word": "chrysanthemum"},
            "id": "call_YpExKmDW7ByjE2dHSxVeIWlX",
            "type": "get_word_length",
        },
    ],
]


def test_partial_json_output_parser() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputToolsParser()

    actual = list(chain.stream(None))
    chunks_without_ids = [
        [{k: v for k, v in tool_call.items() if k != "id"} for tool_call in chunk]
        for chunk in EXPECTED_STREAMED_JSON
    ]
    expected: list = [[]] + chunks_without_ids
    assert actual == expected


async def test_partial_json_output_parser_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputToolsParser()

    actual = [p async for p in chain.astream(None)]
    chunks_without_ids = [
        [{k: v for k, v in tool_call.items() if k != "id"} for tool_call in chunk]
        for chunk in EXPECTED_STREAMED_JSON
    ]
    expected: list = [[]] + chunks_without_ids
    assert actual == expected


def test_partial_json_output_parser_return_id() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputToolsParser(return_id=True)

    actual = list(chain.stream(None))
    assert actual == [[]] + EXPECTED_STREAMED_JSON


def test_partial_json_output_key_parser() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = list(chain.stream(None))
    name_collector_chunks = [
        [tool_call["args"] for tool_call in chunk]
        for chunk in EXPECTED_STREAMED_JSON[:-2]
    ]
    expected: list = [[]] + name_collector_chunks
    assert actual == expected


async def test_partial_json_output_parser_key_async() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputKeyToolsParser(key_name="NameCollector")

    actual = [p async for p in chain.astream(None)]
    name_collector_chunks = [
        [tool_call["args"] for tool_call in chunk]
        for chunk in EXPECTED_STREAMED_JSON[:-2]
    ]
    expected: list = [[]] + name_collector_chunks
    assert actual == expected


def test_partial_json_output_key_parser_first_only() -> None:
    def input_iter(_: Any) -> Iterator[BaseMessage]:
        for msg in STREAMED_MESSAGES:
            yield msg

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )
    expected = [chunk[0]["args"] for chunk in EXPECTED_STREAMED_JSON[:-2]]

    assert list(chain.stream(None)) == expected


async def test_partial_json_output_parser_key_async_first_only() -> None:
    async def input_iter(_: Any) -> AsyncIterator[BaseMessage]:
        for token in STREAMED_MESSAGES:
            yield token

    chain = input_iter | JsonOutputKeyToolsParser(
        key_name="NameCollector", first_tool_only=True
    )
    expected = [chunk[0]["args"] for chunk in EXPECTED_STREAMED_JSON[:-2]]

    assert [p async for p in chain.astream(None)] == expected


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
