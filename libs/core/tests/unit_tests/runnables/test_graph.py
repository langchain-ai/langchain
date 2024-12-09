from typing import Any, Optional

from pydantic import BaseModel
from syrupy import SnapshotAssertion
from typing_extensions import override

from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.xml import XMLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.runnables.graph import Edge, Graph, Node
from langchain_core.runnables.graph_mermaid import _escape_node_label
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION, PYDANTIC_MINOR_VERSION
from tests.unit_tests.pydantic_utils import _normalize_schema


def test_graph_single_runnable(snapshot: SnapshotAssertion) -> None:
    runnable = StrOutputParser()
    graph = StrOutputParser().get_graph()
    first_node = graph.first_node()
    assert first_node is not None
    assert first_node.data.model_json_schema() == runnable.get_input_jsonschema()  # type: ignore[union-attr]
    last_node = graph.last_node()
    assert last_node is not None
    assert last_node.data.model_json_schema() == runnable.get_output_jsonschema()  # type: ignore[union-attr]
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert graph.edges[0].source == first_node.id
    assert graph.edges[1].target == last_node.id
    assert graph.draw_ascii() == snapshot(name="ascii")
    assert graph.draw_mermaid() == snapshot(name="mermaid")

    graph.trim_first_node()
    first_node = graph.first_node()
    assert first_node is not None
    assert first_node.data == runnable

    graph.trim_last_node()
    last_node = graph.last_node()
    assert last_node is not None
    assert last_node.data == runnable


def test_trim(snapshot: SnapshotAssertion) -> None:
    runnable = StrOutputParser()

    class Schema(BaseModel):
        a: int

    graph = Graph()
    start = graph.add_node(Schema, id="__start__")
    ask = graph.add_node(runnable, id="ask_question")
    answer = graph.add_node(runnable, id="answer_question")
    end = graph.add_node(Schema, id="__end__")
    graph.add_edge(start, ask)
    graph.add_edge(ask, answer)
    graph.add_edge(answer, ask, conditional=True)
    graph.add_edge(answer, end, conditional=True)

    assert _normalize_schema(graph.to_json()) == snapshot
    assert graph.first_node() is start
    assert graph.last_node() is end
    # can't trim start or end node
    graph.trim_first_node()
    assert graph.first_node() is start
    graph.trim_last_node()
    assert graph.last_node() is end


def test_trim_multi_edge() -> None:
    class Scheme(BaseModel):
        a: str

    graph = Graph()
    start = graph.add_node(Scheme, id="__start__")
    a = graph.add_node(Scheme, id="a")
    last = graph.add_node(Scheme, id="__end__")

    graph.add_edge(start, a)
    graph.add_edge(a, last)
    graph.add_edge(start, last)

    graph.trim_first_node()  # should not remove __start__ since it has 2 outgoing edges
    assert graph.first_node() is start

    graph.trim_last_node()  # should not remove the __end__ node since it has 2 incoming edges
    assert graph.last_node() is last


def test_graph_sequence(snapshot: SnapshotAssertion) -> None:
    fake_llm = FakeListLLM(responses=["a"])
    prompt = PromptTemplate.from_template("Hello, {name}!")
    list_parser = CommaSeparatedListOutputParser()

    sequence = prompt | fake_llm.with_config(metadata={"key": 2}) | list_parser
    graph = sequence.get_graph()
    assert graph.to_json() == {
        "nodes": [
            {
                "id": 0,
                "type": "schema",
                "data": "PromptInput",
            },
            {
                "id": 1,
                "type": "runnable",
                "data": {
                    "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                    "name": "PromptTemplate",
                },
            },
            {
                "id": 2,
                "type": "runnable",
                "data": {
                    "id": ["langchain_core", "language_models", "fake", "FakeListLLM"],
                    "name": "FakeListLLM",
                },
                "metadata": {"key": 2},
            },
            {
                "id": 3,
                "type": "runnable",
                "data": {
                    "id": [
                        "langchain",
                        "output_parsers",
                        "list",
                        "CommaSeparatedListOutputParser",
                    ],
                    "name": "CommaSeparatedListOutputParser",
                },
            },
            {
                "id": 4,
                "type": "schema",
                "data": "CommaSeparatedListOutputParserOutput",
            },
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 3, "target": 4},
            {"source": 2, "target": 3},
        ],
    }
    assert graph.to_json(with_schemas=True) == {
        "nodes": [
            {
                "id": 0,
                "type": "schema",
                "data": {
                    "title": "PromptInput",
                    "type": "object",
                    "properties": {"name": {"title": "Name", "type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "id": 1,
                "type": "runnable",
                "data": {
                    "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                    "name": "PromptTemplate",
                },
            },
            {
                "id": 2,
                "type": "runnable",
                "data": {
                    "id": ["langchain_core", "language_models", "fake", "FakeListLLM"],
                    "name": "FakeListLLM",
                },
                "metadata": {"key": 2},
            },
            {
                "id": 3,
                "type": "runnable",
                "data": {
                    "id": [
                        "langchain",
                        "output_parsers",
                        "list",
                        "CommaSeparatedListOutputParser",
                    ],
                    "name": "CommaSeparatedListOutputParser",
                },
            },
            {
                "id": 4,
                "type": "schema",
                "data": {
                    "items": {"type": "string"},
                    "title": "CommaSeparatedListOutputParserOutput",
                    "type": "array",
                },
            },
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 3, "target": 4},
            {"source": 2, "target": 3},
        ],
    }
    assert graph.draw_ascii() == snapshot(name="ascii")
    assert graph.draw_mermaid() == snapshot(name="mermaid")


def test_graph_sequence_map(snapshot: SnapshotAssertion) -> None:
    fake_llm = FakeListLLM(responses=["a"])
    prompt = PromptTemplate.from_template("Hello, {name}!")
    list_parser = CommaSeparatedListOutputParser()
    str_parser = StrOutputParser()
    xml_parser = XMLOutputParser()

    def conditional_str_parser(input: str) -> Runnable:
        if input == "a":
            return str_parser
        else:
            return xml_parser

    sequence: Runnable = (
        prompt
        | fake_llm
        | {
            "as_list": list_parser,
            "as_str": conditional_str_parser,
        }
    )
    graph = sequence.get_graph()

    if (PYDANTIC_MAJOR_VERSION, PYDANTIC_MINOR_VERSION) >= (2, 10):
        assert _normalize_schema(graph.to_json(with_schemas=True)) == snapshot(
            name="graph_with_schema"
        )

    if (PYDANTIC_MAJOR_VERSION, PYDANTIC_MINOR_VERSION) >= (2, 10):
        assert _normalize_schema(graph.to_json()) == snapshot(name="graph_no_schemas")

    assert graph.draw_ascii() == snapshot(name="ascii")
    assert graph.draw_mermaid() == snapshot(name="mermaid")
    assert graph.draw_mermaid(with_styles=False) == snapshot(name="mermaid-simple")


def test_parallel_subgraph_mermaid(snapshot: SnapshotAssertion) -> None:
    empty_data = BaseModel
    nodes = {
        "__start__": Node(
            id="__start__", name="__start__", data=empty_data, metadata=None
        ),
        "outer_1": Node(id="outer_1", name="outer_1", data=empty_data, metadata=None),
        "inner_1:inner_1": Node(
            id="inner_1:inner_1", name="inner_1", data=empty_data, metadata=None
        ),
        "inner_1:inner_2": Node(
            id="inner_1:inner_2",
            name="inner_2",
            data=empty_data,
            metadata={"__interrupt": "before"},
        ),
        "inner_2:inner_1": Node(
            id="inner_2:inner_1", name="inner_1", data=empty_data, metadata=None
        ),
        "inner_2:inner_2": Node(
            id="inner_2:inner_2", name="inner_2", data=empty_data, metadata=None
        ),
        "outer_2": Node(id="outer_2", name="outer_2", data=empty_data, metadata=None),
        "__end__": Node(id="__end__", name="__end__", data=empty_data, metadata=None),
    }
    edges = [
        Edge(
            source="inner_1:inner_1",
            target="inner_1:inner_2",
            data=None,
            conditional=False,
        ),
        Edge(
            source="inner_2:inner_1",
            target="inner_2:inner_2",
            data=None,
            conditional=False,
        ),
        Edge(source="__start__", target="outer_1", data=None, conditional=False),
        Edge(
            source="inner_1:inner_2",
            target="outer_2",
            data=None,
            conditional=False,
        ),
        Edge(
            source="inner_2:inner_2",
            target="outer_2",
            data=None,
            conditional=False,
        ),
        Edge(
            source="outer_1",
            target="inner_1:inner_1",
            data=None,
            conditional=False,
        ),
        Edge(
            source="outer_1",
            target="inner_2:inner_1",
            data=None,
            conditional=False,
        ),
        Edge(source="outer_2", target="__end__", data=None, conditional=False),
    ]
    graph = Graph(nodes, edges)
    assert graph.draw_mermaid() == snapshot(name="mermaid")


def test_double_nested_subgraph_mermaid(snapshot: SnapshotAssertion) -> None:
    empty_data = BaseModel
    nodes = {
        "__start__": Node(
            id="__start__", name="__start__", data=empty_data, metadata=None
        ),
        "parent_1": Node(
            id="parent_1", name="parent_1", data=empty_data, metadata=None
        ),
        "child:child_1:grandchild_1": Node(
            id="child:child_1:grandchild_1",
            name="grandchild_1",
            data=empty_data,
            metadata=None,
        ),
        "child:child_1:grandchild_2": Node(
            id="child:child_1:grandchild_2",
            name="grandchild_2",
            data=empty_data,
            metadata={"__interrupt": "before"},
        ),
        "child:child_2": Node(
            id="child:child_2", name="child_2", data=empty_data, metadata=None
        ),
        "parent_2": Node(
            id="parent_2", name="parent_2", data=empty_data, metadata=None
        ),
        "__end__": Node(id="__end__", name="__end__", data=empty_data, metadata=None),
    }
    edges = [
        Edge(
            source="child:child_1:grandchild_1",
            target="child:child_1:grandchild_2",
            data=None,
            conditional=False,
        ),
        Edge(
            source="child:child_1:grandchild_2",
            target="child:child_2",
            data=None,
            conditional=False,
        ),
        Edge(source="__start__", target="parent_1", data=None, conditional=False),
        Edge(
            source="child:child_2",
            target="parent_2",
            data=None,
            conditional=False,
        ),
        Edge(
            source="parent_1",
            target="child:child_1:grandchild_1",
            data=None,
            conditional=False,
        ),
        Edge(source="parent_2", target="__end__", data=None, conditional=False),
    ]
    graph = Graph(nodes, edges)
    assert graph.draw_mermaid() == snapshot(name="mermaid")


def test_runnable_get_graph_with_invalid_input_type() -> None:
    """Test that error isn't raised when getting graph with invalid input type."""

    class InvalidInputTypeRunnable(Runnable[int, int]):
        @property
        @override
        def InputType(self) -> type:
            raise TypeError

        @override
        def invoke(
            self,
            input: int,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
        ) -> int:
            return input

    runnable = InvalidInputTypeRunnable()
    # check whether runnable.invoke works
    assert runnable.invoke(1) == 1
    # check whether runnable.get_graph works
    runnable.get_graph()


def test_runnable_get_graph_with_invalid_output_type() -> None:
    """Test that error is't raised when getting graph with invalid output type."""

    class InvalidOutputTypeRunnable(Runnable[int, int]):
        @property
        @override
        def OutputType(self) -> type:
            raise TypeError

        @override
        def invoke(
            self,
            input: int,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
        ) -> int:
            return input

    runnable = InvalidOutputTypeRunnable()
    # check whether runnable.invoke works
    assert runnable.invoke(1) == 1
    # check whether runnable.get_graph works
    runnable.get_graph()


def test_graph_mermaid_escape_node_label() -> None:
    """Test that node labels are correctly preprocessed for draw_mermaid"""
    assert _escape_node_label("foo") == "foo"
    assert _escape_node_label("foo-bar") == "foo-bar"
    assert _escape_node_label("foo_1") == "foo_1"
    assert _escape_node_label("#foo*&!") == "_foo___"


def test_graph_mermaid_duplicate_nodes(snapshot: SnapshotAssertion) -> None:
    fake_llm = FakeListLLM(responses=["foo", "bar"])
    sequence: Runnable = (
        PromptTemplate.from_template("Hello, {input}")
        | {
            "llm1": fake_llm,
            "llm2": fake_llm,
        }
        | PromptTemplate.from_template("{llm1} {llm2}")
    )
    graph = sequence.get_graph()
    assert graph.draw_mermaid(with_styles=False) == snapshot(name="mermaid")
