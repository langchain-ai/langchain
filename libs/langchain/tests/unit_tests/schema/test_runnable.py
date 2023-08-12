from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest
from freezegun import freeze_time
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langchain import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.chat_models.fake import FakeListChatModel
from langchain.llms.fake import FakeListLLM, FakeStreamingListLLM
from langchain.load.dump import dumpd, dumps
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.document import Document
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    RouterRunnable,
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
    RunnableSequence,
    RunnableWithFallbacks,
)


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution.
    It replaces run ids with deterministic UUIDs for snapshotting."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Run] = []
        self.uuids_map: Dict[UUID, UUID] = {}
        self.uuids_generator = (
            UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
        )

    def _replace_uuid(self, uuid: UUID) -> UUID:
        if uuid not in self.uuids_map:
            self.uuids_map[uuid] = next(self.uuids_generator)
        return self.uuids_map[uuid]

    def _copy_run(self, run: Run) -> Run:
        return run.copy(
            update={
                "id": self._replace_uuid(run.id),
                "parent_run_id": self.uuids_map[run.parent_run_id]
                if run.parent_run_id
                else None,
                "child_runs": [self._copy_run(child) for child in run.child_runs],
                "execution_order": None,
                "child_execution_order": None,
            }
        )

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

        self.runs.append(self._copy_run(run))


class FakeRunnable(Runnable[str, int]):
    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> int:
        return len(input)


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [Document(page_content="foo"), Document(page_content="bar")]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [Document(page_content="foo"), Document(page_content="bar")]


@pytest.mark.asyncio
async def test_default_method_implementations(mocker: MockerFixture) -> None:
    fake = FakeRunnable()
    spy = mocker.spy(fake, "invoke")

    assert fake.invoke("hello", dict(tags=["a-tag"])) == 5
    assert spy.call_args_list == [
        mocker.call("hello", dict(tags=["a-tag"])),
    ]
    spy.reset_mock()

    assert [*fake.stream("hello", dict(metadata={"key": "value"}))] == [5]
    assert spy.call_args_list == [
        mocker.call("hello", dict(metadata={"key": "value"})),
    ]
    spy.reset_mock()

    assert fake.batch(
        ["hello", "wooorld"], [dict(tags=["a-tag"]), dict(metadata={"key": "value"})]
    ) == [5, 7]
    assert spy.call_args_list == [
        mocker.call("hello", dict(tags=["a-tag"])),
        mocker.call("wooorld", dict(metadata={"key": "value"})),
    ]
    spy.reset_mock()

    assert fake.batch(["hello", "wooorld"], dict(tags=["a-tag"])) == [5, 7]
    assert spy.call_args_list == [
        mocker.call("hello", dict(tags=["a-tag"])),
        mocker.call("wooorld", dict(tags=["a-tag"])),
    ]
    spy.reset_mock()

    assert await fake.ainvoke("hello", config={"callbacks": []}) == 5
    assert spy.call_args_list == [
        mocker.call("hello", dict(callbacks=[])),
    ]
    spy.reset_mock()

    assert [part async for part in fake.astream("hello")] == [5]
    assert spy.call_args_list == [
        mocker.call("hello", None),
    ]
    spy.reset_mock()

    assert await fake.abatch(["hello", "wooorld"], dict(metadata={"key": "value"})) == [
        5,
        7,
    ]
    assert spy.call_args_list == [
        mocker.call("hello", dict(metadata={"key": "value"})),
        mocker.call("wooorld", dict(metadata={"key": "value"})),
    ]


@pytest.mark.asyncio
async def test_prompt() -> None:
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    expected = ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )

    assert prompt.invoke({"question": "What is your name?"}) == expected

    assert prompt.batch(
        [
            {"question": "What is your name?"},
            {"question": "What is your favorite color?"},
        ]
    ) == [
        expected,
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your favorite color?"),
            ]
        ),
    ]

    assert [*prompt.stream({"question": "What is your name?"})] == [expected]

    assert await prompt.ainvoke({"question": "What is your name?"}) == expected

    assert await prompt.abatch(
        [
            {"question": "What is your name?"},
            {"question": "What is your favorite color?"},
        ]
    ) == [
        expected,
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your favorite color?"),
            ]
        ),
    ]

    assert [
        part async for part in prompt.astream({"question": "What is your name?"})
    ] == [expected]


@pytest.mark.asyncio
@freeze_time("2023-01-01")
async def test_prompt_with_chat_model(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo"])

    chain = prompt | chat

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == []
    assert chain.last == chat
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == AIMessage(content="foo")
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert tracer.runs == snapshot
    mocker.stop(prompt_spy)
    mocker.stop(chat_spy)

    # Test batch
    prompt_spy = mocker.spy(prompt.__class__, "batch")
    chat_spy = mocker.spy(chat.__class__, "batch")
    tracer = FakeTracer()
    assert chain.batch(
        [
            {"question": "What is your name?"},
            {"question": "What is your favorite color?"},
        ],
        dict(callbacks=[tracer]),
    ) == [
        AIMessage(content="foo"),
        AIMessage(content="foo"),
    ]
    assert prompt_spy.call_args.args[1] == [
        {"question": "What is your name?"},
        {"question": "What is your favorite color?"},
    ]
    assert chat_spy.call_args.args[1] == [
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your name?"),
            ]
        ),
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your favorite color?"),
            ]
        ),
    ]
    assert (
        len(
            [
                r
                for r in tracer.runs
                if r.parent_run_id is None and len(r.child_runs) == 2
            ]
        )
        == 2
    ), "Each of 2 outer runs contains exactly two inner runs (1 prompt, 1 chat)"
    mocker.stop(prompt_spy)
    mocker.stop(chat_spy)

    # Test stream
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "stream")
    tracer = FakeTracer()
    assert [
        *chain.stream({"question": "What is your name?"}, dict(callbacks=[tracer]))
    ] == [AIMessage(content="f"), AIMessage(content="o"), AIMessage(content="o")]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )


@pytest.mark.asyncio
@freeze_time("2023-01-01")
async def test_prompt_with_llm(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeListLLM(responses=["foo", "bar"])

    chain = prompt | llm

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == []
    assert chain.last == llm
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    llm_spy = mocker.spy(llm.__class__, "ainvoke")
    tracer = FakeTracer()
    assert (
        await chain.ainvoke(
            {"question": "What is your name?"}, dict(callbacks=[tracer])
        )
        == "foo"
    )
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert tracer.runs == snapshot
    mocker.stop(prompt_spy)
    mocker.stop(llm_spy)

    # Test batch
    prompt_spy = mocker.spy(prompt.__class__, "abatch")
    llm_spy = mocker.spy(llm.__class__, "abatch")
    tracer = FakeTracer()
    assert await chain.abatch(
        [
            {"question": "What is your name?"},
            {"question": "What is your favorite color?"},
        ],
        dict(callbacks=[tracer]),
    ) == ["bar", "foo"]
    assert prompt_spy.call_args.args[1] == [
        {"question": "What is your name?"},
        {"question": "What is your favorite color?"},
    ]
    assert llm_spy.call_args.args[1] == [
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your name?"),
            ]
        ),
        ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your favorite color?"),
            ]
        ),
    ]
    assert tracer.runs == snapshot
    mocker.stop(prompt_spy)
    mocker.stop(llm_spy)

    # Test stream
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    llm_spy = mocker.spy(llm.__class__, "astream")
    tracer = FakeTracer()
    assert [
        token
        async for token in chain.astream(
            {"question": "What is your name?"}, dict(callbacks=[tracer])
        )
    ] == ["bar"]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )


@freeze_time("2023-01-01")
def test_prompt_with_chat_model_and_parser(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo, bar"])
    parser = CommaSeparatedListOutputParser()

    chain = prompt | chat | parser

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [chat]
    assert chain.last == parser
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    parser_spy = mocker.spy(parser.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == ["foo", "bar"]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert parser_spy.call_args.args[1] == AIMessage(content="foo, bar")
    assert tracer.runs == snapshot


@freeze_time("2023-01-01")
def test_combining_sequences(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo, bar"])
    parser = CommaSeparatedListOutputParser()

    chain = prompt | chat | parser

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [chat]
    assert chain.last == parser
    assert dumps(chain, pretty=True) == snapshot

    prompt2 = (
        SystemMessagePromptTemplate.from_template("You are a nicer assistant.")
        + "{question}"
    )
    chat2 = FakeListChatModel(responses=["baz, qux"])
    parser2 = CommaSeparatedListOutputParser()
    input_formatter: RunnableLambda[List[str], Dict[str, Any]] = RunnableLambda(
        lambda x: {"question": x[0] + x[1]}
    )

    chain2 = input_formatter | prompt2 | chat2 | parser2

    assert isinstance(chain, RunnableSequence)
    assert chain2.first == input_formatter
    assert chain2.middle == [prompt2, chat2]
    assert chain2.last == parser2
    assert dumps(chain2, pretty=True) == snapshot

    combined_chain = chain | chain2

    assert combined_chain.first == prompt
    assert combined_chain.middle == [
        chat,
        parser,
        input_formatter,
        prompt2,
        chat2,
    ]
    assert combined_chain.last == parser2
    assert dumps(combined_chain, pretty=True) == snapshot

    # Test invoke
    tracer = FakeTracer()
    assert combined_chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == ["baz", "qux"]
    assert tracer.runs == snapshot


@freeze_time("2023-01-01")
def test_seq_dict_prompt_llm(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    passthrough = mocker.Mock(side_effect=lambda x: x)

    retriever = FakeRetriever()

    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + """Context:
{documents}

Question:
{question}"""
    )

    chat = FakeListChatModel(responses=["foo, bar"])

    parser = CommaSeparatedListOutputParser()

    chain = (
        {
            "question": RunnablePassthrough[str]() | passthrough,
            "documents": passthrough | retriever,
            "just_to_test_lambda": passthrough,
        }
        | prompt
        | chat
        | parser
    )

    assert isinstance(chain, RunnableSequence)
    assert isinstance(chain.first, RunnableMap)
    assert chain.middle == [prompt, chat]
    assert chain.last == parser
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    parser_spy = mocker.spy(parser.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke("What is your name?", dict(callbacks=[tracer])) == [
        "foo",
        "bar",
    ]
    assert prompt_spy.call_args.args[1] == {
        "documents": [Document(page_content="foo"), Document(page_content="bar")],
        "question": "What is your name?",
        "just_to_test_lambda": "What is your name?",
    }
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(
                content="""Context:
[Document(page_content='foo', metadata={}), Document(page_content='bar', metadata={})]

Question:
What is your name?"""
            ),
        ]
    )
    assert parser_spy.call_args.args[1] == AIMessage(content="foo, bar")
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 4
    map_run = parent_run.child_runs[0]
    assert map_run.name == "RunnableMap"
    assert len(map_run.child_runs) == 3


@freeze_time("2023-01-01")
def test_seq_prompt_dict(mocker: MockerFixture, snapshot: SnapshotAssertion) -> None:
    passthrough = mocker.Mock(side_effect=lambda x: x)

    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat = FakeListChatModel(responses=["i'm a chatbot"])

    llm = FakeListLLM(responses=["i'm a textbot"])

    chain = (
        prompt
        | passthrough
        | {
            "chat": chat,
            "llm": llm,
        }
    )

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [RunnableLambda(passthrough)]
    assert isinstance(chain.last, RunnableMap)
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    llm_spy = mocker.spy(llm.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == {
        "chat": AIMessage(content="i'm a chatbot"),
        "llm": "i'm a textbot",
    }
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 3
    map_run = parent_run.child_runs[2]
    assert map_run.name == "RunnableMap"
    assert len(map_run.child_runs) == 2


@pytest.mark.asyncio
@freeze_time("2023-01-01")
async def test_router_runnable(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    chain1 = ChatPromptTemplate.from_template(
        "You are a math genius. Answer the question: {question}"
    ) | FakeListLLM(responses=["4"])
    chain2 = ChatPromptTemplate.from_template(
        "You are an english major. Answer the question: {question}"
    ) | FakeListLLM(responses=["2"])
    router = RouterRunnable({"math": chain1, "english": chain2})
    chain: Runnable = {
        "key": lambda x: x["key"],
        "input": {"question": lambda x: x["question"]},
    } | router
    assert dumps(chain, pretty=True) == snapshot

    result = chain.invoke({"key": "math", "question": "2 + 2"})
    assert result == "4"

    result2 = chain.batch(
        [{"key": "math", "question": "2 + 2"}, {"key": "english", "question": "2 + 2"}]
    )
    assert result2 == ["4", "2"]

    result = await chain.ainvoke({"key": "math", "question": "2 + 2"})
    assert result == "4"

    result2 = await chain.abatch(
        [{"key": "math", "question": "2 + 2"}, {"key": "english", "question": "2 + 2"}]
    )
    assert result2 == ["4", "2"]

    # Test invoke
    router_spy = mocker.spy(router.__class__, "invoke")
    tracer = FakeTracer()
    assert (
        chain.invoke({"key": "math", "question": "2 + 2"}, dict(callbacks=[tracer]))
        == "4"
    )
    assert router_spy.call_args.args[1] == {
        "key": "math",
        "input": {"question": "2 + 2"},
    }
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 2
    router_run = parent_run.child_runs[1]
    assert router_run.name == "RunnableSequence"  # TODO: should be RunnableRouter
    assert len(router_run.child_runs) == 2


@freeze_time("2023-01-01")
def test_seq_prompt_map(mocker: MockerFixture, snapshot: SnapshotAssertion) -> None:
    passthrough = mocker.Mock(side_effect=lambda x: x)

    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat = FakeListChatModel(responses=["i'm a chatbot"])

    llm = FakeListLLM(responses=["i'm a textbot"])

    chain = (
        prompt
        | passthrough
        | {
            "chat": chat.bind(stop=["Thought:"]),
            "llm": llm,
            "passthrough": passthrough,
        }
    )

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [RunnableLambda(passthrough)]
    assert isinstance(chain.last, RunnableMap)
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "invoke")
    chat_spy = mocker.spy(chat.__class__, "invoke")
    llm_spy = mocker.spy(llm.__class__, "invoke")
    tracer = FakeTracer()
    assert chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == {
        "chat": AIMessage(content="i'm a chatbot"),
        "llm": "i'm a textbot",
        "passthrough": ChatPromptValue(
            messages=[
                SystemMessage(content="You are a nice assistant."),
                HumanMessage(content="What is your name?"),
            ]
        ),
    }
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 3
    map_run = parent_run.child_runs[2]
    assert map_run.name == "RunnableMap"
    assert len(map_run.child_runs) == 3


def test_bind_bind() -> None:
    llm = FakeListLLM(responses=["i'm a textbot"])

    assert dumpd(
        llm.bind(stop=["Thought:"], one="two").bind(
            stop=["Observation:"], hello="world"
        )
    ) == dumpd(llm.bind(stop=["Observation:"], one="two", hello="world"))


def test_deep_stream() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain = prompt | llm | StrOutputParser()

    stream = chain.stream({"question": "What up"})

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"

    chunks = []
    for chunk in (chain | RunnablePassthrough()).stream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"


@pytest.mark.asyncio
async def test_deep_astream() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain = prompt | llm | StrOutputParser()

    stream = chain.astream({"question": "What up"})

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"

    chunks = []
    async for chunk in (chain | RunnablePassthrough()).astream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"


@pytest.fixture()
def llm_with_fallbacks() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([pass_llm])


@pytest.fixture()
def llm_with_multi_fallbacks() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    error_llm_2 = FakeListLLM(responses=["baz"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([error_llm_2, pass_llm])


@pytest.fixture()
def llm_chain_with_fallbacks() -> RunnableSequence:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    prompt = PromptTemplate.from_template("what did baz say to {buz}")
    return RunnableMap({"buz": lambda x: x}) | (prompt | error_llm).with_fallbacks(
        [prompt | pass_llm]
    )


@pytest.mark.parametrize(
    "runnable",
    ["llm_with_fallbacks", "llm_with_multi_fallbacks", "llm_chain_with_fallbacks"],
)
@pytest.mark.asyncio
async def test_llm_with_fallbacks(
    runnable: RunnableWithFallbacks, request: Any, snapshot: SnapshotAssertion
) -> None:
    runnable = request.getfixturevalue(runnable)
    assert runnable.invoke("hello") == "bar"
    assert runnable.batch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(runnable.stream("hello")) == ["bar"]
    assert await runnable.ainvoke("hello") == "bar"
    assert await runnable.abatch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(await runnable.ainvoke("hello")) == list("bar")
    assert dumps(runnable, pretty=True) == snapshot
