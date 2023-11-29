import pytest

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompt_values import StringPromptValue
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.context import ContextGet, ContextSet
from langchain_core.runnables.passthrough import RunnablePassthrough
from tests.unit_tests.fake.llm import FakeListLLM


def test_runnable_context_seq_basic_get_set() -> None:
    seq = ContextSet("foo") | ContextGet("foo")

    result = seq.invoke("foo")
    assert result == "foo"


def test_runnable_context_seq_get_in_map() -> None:
    seq = ContextSet("input") | {"bar": ContextGet("input")}

    assert seq.invoke("foo") == {"bar": "foo"}


def test_runnable_context_seq_put_in_map() -> None:
    seq = {"bar": ContextSet("input")} | ContextGet("input")

    assert seq.invoke("foo") == "foo"


def test_runnable_context_seq_key_not_found() -> None:
    seq = {"bar": ContextSet("input")} | ContextGet("foo")

    with pytest.raises(KeyError):
        seq.invoke("foo")


def test_runnable_context_seq_invoke() -> None:
    prompt = PromptTemplate.from_template("{foo} {bar}")
    llm = FakeListLLM(responses=["hello"])

    seq = (
        prompt
        | ContextSet("prompt")
        | llm
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    result = seq.invoke({"foo": "foo", "bar": "bar"})
    assert result == {"response": "hello", "prompt": StringPromptValue(text="foo bar")}


@pytest.mark.asyncio
async def test_runnable_context_seq_ainvoke() -> None:
    prompt = PromptTemplate.from_template("{foo} {bar}")
    llm = FakeListLLM(responses=["hello"])

    seq = (
        prompt
        | ContextSet("prompt")
        | llm
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    result = await seq.ainvoke({"foo": "foo", "bar": "bar"})
    assert result == {"response": "hello", "prompt": StringPromptValue(text="foo bar")}


def test_runnable_context_seq_batch() -> None:
    prompt = PromptTemplate.from_template("{foo} {bar}")
    llm = FakeListLLM(responses=["hello"])

    seq = (
        prompt
        | ContextSet("prompt")
        | llm
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    result = seq.batch(
        [
            {"foo": "foo", "bar": "bar"},
            {"foo": "bar", "bar": "foo"},
            {"foo": "a", "bar": "b"},
        ]
    )
    assert result == [
        {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
        {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
        {"response": "hello", "prompt": StringPromptValue(text="a b")},
    ]


@pytest.mark.asyncio
async def test_runnable_context_seq_abatch() -> None:
    prompt = PromptTemplate.from_template("{foo} {bar}")
    llm = FakeListLLM(responses=["hello"])

    seq = (
        prompt
        | ContextSet("prompt")
        | llm
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    result = await seq.abatch(
        [
            {"foo": "foo", "bar": "bar"},
            {"foo": "bar", "bar": "foo"},
            {"foo": "a", "bar": "b"},
        ]
    )
    assert result == [
        {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
        {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
        {"response": "hello", "prompt": StringPromptValue(text="a b")},
    ]


@pytest.mark.asyncio
async def test_runnable_context_seq_decorator() -> None:
    prompt = PromptTemplate.from_template("{foo} {bar}")
    llm = FakeListLLM(responses=["hello"])

    mock_llm_chain = (
        prompt
        | ContextSet("prompt")
        | llm
        | StrOutputParser()
        | {
            "response": RunnablePassthrough(),
            "prompt": ContextGet("prompt"),
        }
    )

    result = mock_llm_chain.invoke({"foo": "foo", "bar": "bar"})
    assert result == {"response": "hello", "prompt": StringPromptValue(text="foo bar")}

    result = await mock_llm_chain.ainvoke({"foo": "foo", "bar": "bar"})
    assert result == {"response": "hello", "prompt": StringPromptValue(text="foo bar")}

    result = mock_llm_chain.batch(
        [
            {"foo": "foo", "bar": "bar"},
            {"foo": "bar", "bar": "foo"},
        ]
    )
    assert result == [
        {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
        {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
    ]

    result = await mock_llm_chain.abatch(
        [
            {"foo": "foo", "bar": "bar"},
            {"foo": "bar", "bar": "foo"},
        ]
    )
    assert result == [
        {"response": "hello", "prompt": StringPromptValue(text="foo bar")},
        {"response": "hello", "prompt": StringPromptValue(text="bar foo")},
    ]


def test_runnable_context_seq_naive_rag() -> None:
    context = [
        "Hi there!",
        "How are you?",
        "What's your name?",
    ]

    retriever = RunnableLambda(lambda x: context)
    prompt = PromptTemplate.from_template("{context} {question}")
    llm = FakeListLLM(responses=["hello"])

    chain = (
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
    result = chain.invoke("What up")
    assert result == {
        "result": "hello",
        "context": context,
        "input": "What up",
    }
