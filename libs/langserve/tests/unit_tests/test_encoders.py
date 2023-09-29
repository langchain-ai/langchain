import json
from typing import Any

import pytest
from langchain.schema.messages import (
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel

from langserve.serialization import simple_dumps, simple_loads


@pytest.mark.parametrize(
    "data, expected_json",
    [
        # Test with python primitives
        (1, 1),
        ([], []),
        ({}, {}),
        ({"a": 1}, {"a": 1}),
        (
            {"output": [HumanMessage(content="hello")]},
            {
                "output": [
                    {
                        "content": "hello",
                        "additional_kwargs": {},
                        "type": "human",
                        "example": False,
                        "is_chunk": False,
                    }
                ]
            },
        ),
        # Test with a single message (HumanMessage)
        (
            HumanMessage(content="Hello"),
            {
                "additional_kwargs": {},
                "content": "Hello",
                "example": False,
                "type": "human",
                "is_chunk": False,
            },
        ),
        # Test with a list containing mixed elements
        (
            [HumanMessage(content="Hello"), SystemMessage(content="Hi"), 42, "world"],
            [
                {
                    "additional_kwargs": {},
                    "content": "Hello",
                    "example": False,
                    "type": "human",
                    "is_chunk": False,
                },
                {
                    "additional_kwargs": {},
                    "content": "Hi",
                    "type": "system",
                    "is_chunk": False,
                },
                42,
                "world",
            ],
        ),
        # Attention: This test is not correct right now
        # Test with full and chunk messages
        (
            [HumanMessage(content="Hello"), HumanMessageChunk(content="Hi")],
            [
                {
                    "additional_kwargs": {},
                    "content": "Hello",
                    "example": False,
                    "type": "human",
                    "is_chunk": False,
                },
                {
                    "additional_kwargs": {},
                    "content": "Hi",
                    "example": False,
                    "type": "human",
                    "is_chunk": True,
                },
            ],
        ),
        # Attention: This test is not correct right now
        # Test with full and chunk messages
        (
            [HumanMessageChunk(content="Hello"), HumanMessage(content="Hi")],
            [
                {
                    "additional_kwargs": {},
                    "content": "Hello",
                    "example": False,
                    "type": "human",
                    "is_chunk": True,
                },
                {
                    "additional_kwargs": {},
                    "content": "Hi",
                    "example": False,
                    "type": "human",
                    "is_chunk": False,
                },
            ],
        ),
        # Test with a dictionary containing mixed elements
        (
            {
                "message": HumanMessage(content="Greetings"),
                "numbers": [1, 2, 3],
                "boom": "Hello, world!",
            },
            {
                "message": {
                    "additional_kwargs": {},
                    "content": "Greetings",
                    "example": False,
                    "type": "human",
                    "is_chunk": False,
                },
                "numbers": [1, 2, 3],
                "boom": "Hello, world!",
            },
        ),
    ],
)
def test_serialization(data: Any, expected_json: Any) -> None:
    """Test that the LangChainEncoder encodes the data as expected."""
    # Test encoding
    assert json.loads(simple_dumps(data)) == expected_json
    # Test decoding
    assert simple_loads(json.dumps(expected_json)) == data
    # Test full representation are equivalent including the pydantic model classes
    assert _get_full_representation(data) == _get_full_representation(
        simple_loads(json.dumps(expected_json))
    )


def _get_full_representation(data: Any) -> Any:
    """Get the full representation of the data, replacing pydantic models with schema.

    Pydantic tests two different models for equality based on equality
    of their schema; instead we will rely on the equality of their full
    schema representation. This will make sure that both models have the
    same name (e.g., HumanMessage vs. HumanMessageChunk).

    Args:
        data: python primitives + pydantic models

    Returns:
        data represented entirely with python primitives
    """
    if isinstance(data, dict):
        return {key: _get_full_representation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_get_full_representation(value) for value in data]
    elif isinstance(data, BaseModel):
        return data.schema()
    else:
        return data
