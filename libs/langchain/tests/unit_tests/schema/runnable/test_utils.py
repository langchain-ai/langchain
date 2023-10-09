import itertools
import sys
from typing import Any, Callable, Iterable, Iterator

import pytest

from langchain.schema.messages import AIMessageChunk, HumanMessageChunk
from langchain.schema.output import ChatGenerationChunk
from langchain.schema.runnable.utils import (
    AddableDict,
    RunnableStreamResetMarker,
    add,
    get_lambda_source,
    indent_lines_after_first,
)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
@pytest.mark.parametrize(
    "func, expected_source",
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),
    ],
)
def test_get_lambda_source(func: Callable, expected_source: str) -> None:
    """Test get_lambda_source function"""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    "text,prefix,expected_output",
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function"""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output


def roundrobin(*iterables: Iterable[Any]) -> Iterator[Any]:
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def test_addable_dict() -> None:
    gen_chunks = [
        AddableDict({"gen": chunk})
        for chunk in [
            ChatGenerationChunk(message=HumanMessageChunk(content="Hello, ")),
            RunnableStreamResetMarker(),
            ChatGenerationChunk(
                message=HumanMessageChunk(content="world!"),
                generation_info={"foo": "bar"},
            ),
            ChatGenerationChunk(
                message=HumanMessageChunk(content="!"), generation_info={"baz": "foo"}
            ),
        ]
    ]
    message_chunks = [
        AddableDict({"msg": chunk})
        for chunk in [
            AIMessageChunk(
                content="", additional_kwargs={"function_call": {"name": "web_search"}}
            ),
            RunnableStreamResetMarker(),
            AIMessageChunk(
                content="", additional_kwargs={"function_call": {"arguments": "{\n"}}
            ),
            AIMessageChunk(
                content="",
                additional_kwargs={
                    "function_call": {"arguments": '  "query": "turtles"\n}'}
                },
            ),
        ]
    ]

    final = add(roundrobin(gen_chunks, message_chunks))

    assert final == AddableDict(
        {
            "gen": ChatGenerationChunk(
                message=HumanMessageChunk(content="world!!"),
                generation_info={"foo": "bar", "baz": "foo"},
            ),
            "msg": AIMessageChunk(
                content="",
                additional_kwargs={
                    "function_call": {
                        "arguments": '{\n  "query": "turtles"\n}',
                    }
                },
            ),
        }
    )

    final = add(itertools.chain(gen_chunks, message_chunks))

    assert final == AddableDict(
        {
            "gen": ChatGenerationChunk(
                message=HumanMessageChunk(content="world!!"),
                generation_info={"foo": "bar", "baz": "foo"},
            ),
            "msg": AIMessageChunk(
                content="",
                additional_kwargs={
                    "function_call": {
                        "arguments": '{\n  "query": "turtles"\n}',
                    }
                },
            ),
        }
    )
