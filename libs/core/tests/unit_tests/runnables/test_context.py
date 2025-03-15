import asyncio
from typing import Any, Callable, NamedTuple, Union

import pytest

from langchain_core.beta.runnables.context import Context
from langchain_core.language_models import FakeListLLM, FakeStreamingListLLM
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import Runnable, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import aadd, add


class _TestCase(NamedTuple):
    input: Any
    output: Any


def seq_naive_rag() -> Runnable:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    return (
        Context.setter("input")
        | {
            "context": retriever | Context.setter("context"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | {
            "result": RunnablePassthrough(),
            "context": Context.getter("context"),
            "input": Context.getter("input"),
        }
    )


def seq_naive_rag_alt() -> Runnable:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    return (
        Context.setter("input")
        | {
            "context": retriever | Context.setter("context"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | Context.setter("result")
        | Context.getter(["context", "input", "result"])
    )


def seq_naive_rag_scoped() -> Runnable:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    scoped = Context.create_scope("a_scope")

    return (
        Context.setter("input")
        | {
            "context": retriever | Context.setter("context"),
            "question": RunnablePassthrough(),
            "scoped": scoped.setter("context") | scoped.getter("context"),
        }
        | prompt
        | llm
        | StrOutputParser()
        | Context.setter("result")
        | Context.getter(["context", "input", "result"])
    )


test_cases = [
    (
        Context.setter("foo") | Context.getter("foo"),
        (
            _TestCase("foo", "foo"),
            _TestCase("bar", "bar"),
        ),
    ),
    (
        Context.setter("input") | {"bar": Context.getter("input")},
        (
            _TestCase("foo", {"bar": "foo"}),
            _TestCase("bar", {"bar": "bar"}),
        ),
    ),
    (
        {"bar": Context.setter("input")} | Context.getter("input"),
        (
            _TestCase("foo", "foo"),
            _TestCase("bar", "bar"),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | Context.setter("prompt")
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": Context.getter("prompt"),
            }
        ),
        (
            _TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
            ),
            _TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | Context.setter("prompt", prompt_str=lambda x: x.to_string())
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": Context.getter("prompt"),
                "prompt_str": Context.getter("prompt_str"),
            }
        ),
        (
            _TestCase(
                {"foo": "foo", "bar": "bar"},
                {
                    "response": "hello",
                    "prompt": StringPromptValue(text="foo bar"),
                    "prompt_str": "foo bar",
                },
            ),
            _TestCase(
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
            | Context.setter(prompt_str=lambda x: x.to_string())
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt_str": Context.getter("prompt_str"),
            }
        ),
        (
            _TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt_str": "foo bar"},
            ),
            _TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt_str": "bar foo"},
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | Context.setter("prompt_str", lambda x: x.to_string())
            | FakeListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt_str": Context.getter("prompt_str"),
            }
        ),
        (
            _TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt_str": "foo bar"},
            ),
            _TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt_str": "bar foo"},
            ),
        ),
    ),
    (
        (
            PromptTemplate.from_template("{foo} {bar}")
            | Context.setter("prompt")
            | FakeStreamingListLLM(responses=["hello"])
            | StrOutputParser()
            | {
                "response": RunnablePassthrough(),
                "prompt": Context.getter("prompt"),
            }
        ),
        (
            _TestCase(
                {"foo": "foo", "bar": "bar"},
                {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
            ),
            _TestCase(
                {"foo": "bar", "bar": "foo"},
                {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
            ),
        ),
    ),
    (
        seq_naive_rag,
        (
            _TestCase(
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
            _TestCase(
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
            _TestCase(
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
            _TestCase(
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
        seq_naive_rag_scoped,
        (
            _TestCase(
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
            _TestCase(
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
def test_context_runnables(
    runnable: Union[Runnable, Callable[[], Runnable]], cases: list[_TestCase]
) -> None:
    runnable = runnable if isinstance(runnable, Runnable) else runnable()
    assert runnable.invoke(cases[0].input) == cases[0].output
    assert runnable.batch([case.input for case in cases]) == [
        case.output for case in cases
    ]
    assert add(runnable.stream(cases[0].input)) == cases[0].output


@pytest.mark.parametrize("runnable, cases", test_cases)
async def test_context_runnables_async(
    runnable: Union[Runnable, Callable[[], Runnable]], cases: list[_TestCase]
) -> None:
    runnable = runnable if isinstance(runnable, Runnable) else runnable()
    assert await runnable.ainvoke(cases[1].input) == cases[1].output
    assert await runnable.abatch([case.input for case in cases]) == [
        case.output for case in cases
    ]
    assert await aadd(runnable.astream(cases[1].input)) == cases[1].output


def test_runnable_context_seq_key_not_found() -> None:
    seq: Runnable = {"bar": Context.setter("input")} | Context.getter("foo")

    with pytest.raises(ValueError):
        seq.invoke("foo")


def test_runnable_context_seq_key_order() -> None:
    seq: Runnable = {"bar": Context.getter("foo")} | Context.setter("foo")

    with pytest.raises(ValueError):
        seq.invoke("foo")


def test_runnable_context_deadlock() -> None:
    seq: Runnable = {
        "bar": Context.setter("input") | Context.getter("foo"),
        "foo": Context.setter("foo") | Context.getter("input"),
    } | RunnablePassthrough()

    with pytest.raises(ValueError):
        seq.invoke("foo")


def test_runnable_context_seq_key_circular_ref() -> None:
    seq: Runnable = {
        "bar": Context.setter(input=Context.getter("input"))
    } | Context.getter("foo")

    with pytest.raises(ValueError):
        seq.invoke("foo")


async def test_runnable_seq_streaming_chunks() -> None:
    chain: Runnable = (
        PromptTemplate.from_template("{foo} {bar}")
        | Context.setter("prompt")
        | FakeStreamingListLLM(responses=["hello"])
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": Context.getter("prompt"),
        }
    )
    chunks = await asyncio.to_thread(list, chain.stream({"foo": "foo", "bar": "bar"}))
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
