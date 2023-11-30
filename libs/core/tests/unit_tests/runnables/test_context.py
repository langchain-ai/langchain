from typing import Any, Callable, List, NamedTuple, Union

import pytest

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.runnables.context import ContextGet, ContextSet
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import aadd, add
from tests.unit_tests.fake.llm import FakeListLLM, FakeStreamingListLLM


class TestCase(NamedTuple):
    input: Any
    output: Any


def seq_naive_rag() -> None:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    return (
        ContextSet("input")
        | {
            "context": retriever | ContextSet("context"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | {
            "result": RunnablePassthrough(),
            "context": ContextGet("context"),
            "input": ContextGet("input"),
        }
    )


def seq_naive_rag_alt() -> None:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    return (
        ContextSet("input")
        | {
            "context": retriever | ContextSet("context"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | ContextSet("result")
        | ContextGet(["context", "input", "result"])
    )


test_cases = [
    (
        ContextSet("foo") | ContextGet("foo"),
        (
            TestCase("foo", "foo"),
            TestCase("bar", "bar"),
        ),
    ),
    (
        ContextSet("input") | {"bar": ContextGet("input")},
        (
            TestCase("foo", {"bar": "foo"}),
            TestCase("bar", {"bar": "bar"}),
        ),
    ),
    (
        {"bar": ContextSet("input")} | ContextGet("input"),
        (
            TestCase("foo", "foo"),
            TestCase("bar", "bar"),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | ContextSet("prompt")
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": ContextGet("prompt"),
            }
        ),
        (
            TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
            ),
            TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | ContextSet("prompt", prompt_str=lambda x: x.to_string())
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": ContextGet("prompt"),
                "prompt_str": ContextGet("prompt_str"),
            }
        ),
        (
            TestCase(
                {"foo": "foo", "bar": "bar"},
                {
                    "response": "hello",
                    "prompt": StringPromptValue(text="foo bar"),
                    "prompt_str": "foo bar",
                },
            ),
            TestCase(
                {"foo": "bar", "bar": "foo"},
                {
                    "response": "hello",
                    "prompt": StringPromptValue(text="bar foo"),
                    "prompt_str": "bar foo",
                },
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | ContextSet("prompt_str", lambda x: x.to_string())
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt_str": ContextGet("prompt_str"),
            }
        ),
        (
            TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt_str": "foo bar"},
            ),
            TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt_str": "bar foo"},
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | ContextSet("prompt")
            | FakeStreamingListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": ContextGet("prompt"),
            }
        ),
        (
            TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
            ),
            TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
            ),
        ),
    ),
    (
        seq_naive_rag,
        (
            TestCase(
                "What up",
                {
                    "result": "hello",
                    "context": [
                        "Hi there!",
                        "How are you?",
                        "What's your name?",
                    ],
                    "input": "What up",
                },
            ),
            TestCase(
                "Howdy",
                {
                    "result": "hello",
                    "context": [
                        "Hi there!",
                        "How are you?",
                        "What's your name?",
                    ],
                    "input": "Howdy",
                },
            ),
        ),
    ),
    (
        seq_naive_rag_alt,
        (
            TestCase(
                "What up",
                {
                    "result": "hello",
                    "context": [
                        "Hi there!",
                        "How are you?",
                        "What's your name?",
                    ],
                    "input": "What up",
                },
            ),
            TestCase(
                "Howdy",
                {
                    "result": "hello",
                    "context": [
                        "Hi there!",
                        "How are you?",
                        "What's your name?",
                    ],
                    "input": "Howdy",
                },
            ),
        ),
    ),
]


@pytest.mark.parametrize("runnable, cases", test_cases)
async def test_context_runnables(
    runnable: Union[Runnable, Callable[[], Runnable]], cases: List[TestCase]
):
    runnable = runnable if isinstance(runnable, Runnable) else runnable()
    assert runnable.invoke(cases[0].input) == cases[0].output
    assert await runnable.ainvoke(cases[1].input) == cases[1].output
    assert runnable.batch([case.input for case in cases]) == [
        case.output for case in cases
    ]
    assert await runnable.abatch([case.input for case in cases]) == [
        case.output for case in cases
    ]
    assert add(runnable.stream(cases[0].input)) == cases[0].output
    assert await aadd(runnable.astream(cases[1].input)) == cases[1].output


def test_runnable_context_seq_key_not_found() -> None:
    seq = {"bar": ContextSet("input")} | ContextGet("foo")

    with pytest.raises(KeyError):
        seq.invoke("foo")


def test_runnable_context_seq_key_circular_ref() -> None:
    seq = {"bar": ContextSet(input=ContextGet("input"))} | ContextGet("foo")

    with pytest.raises(ValueError):
        seq.invoke("foo")


async def test_runnable_seq_streaming_chunks() -> None:
    chain = (
        PromptTemplate.from_template("{foo} {bar}")
        | ContextSet("prompt")
        | FakeStreamingListLLM(responses=["hello"])
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    chunks = [c for c in chain.stream({"foo": "foo", "bar": "bar"})]
    achunks = [c async for c in chain.astream({"foo": "foo", "bar": "bar"})]
    for c in chunks:
        assert c in achunks
    for c in achunks:
        assert c in chunks

    assert len(chunks) == 6
    assert [c for c in chunks if c.get("response")] == [
        {"response": "h"},
        {"response": "e"},
        {"response": "l"},
        {"response": "l"},
        {"response": "o"},
    ]
    assert [c for c in chunks if c.get("prompt")] == [
        {"prompt": StringPromptValue(text="foo bar")},
    ]
