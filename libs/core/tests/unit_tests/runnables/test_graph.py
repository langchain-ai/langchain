from typing import Optional

from syrupy import SnapshotAssertion

from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.xml import XMLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.graph_mermaid import _escape_node_label


def test_graph_single_runnable(snapshot: SnapshotAssertion) -> None:
    runnable = StrOutputParser()
    graph = StrOutputParser().get_graph()
    first_node = graph.first_node()
    assert first_node is not None
    assert first_node.data.schema() == runnable.input_schema.schema()  # type: ignore[union-attr]
    last_node = graph.last_node()
    assert last_node is not None
    assert last_node.data.schema() == runnable.output_schema.schema()  # type: ignore[union-attr]
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

    assert graph.to_json() == snapshot
    assert graph.first_node() is start
    assert graph.last_node() is end
    # can't trim start or end node
    graph.trim_first_node()
    assert graph.first_node() is start
    graph.trim_last_node()
    assert graph.last_node() is end


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
    assert graph.to_json(with_schemas=True) == snapshot(name="graph_with_schema")
    assert graph.to_json() == snapshot(name="graph_no_schemas")
    assert graph.draw_ascii() == snapshot(name="ascii")
    assert graph.draw_mermaid() == snapshot(name="mermaid")
    assert graph.draw_mermaid(with_styles=False) == snapshot(name="mermaid-simple")


def test_runnable_get_graph_with_invalid_input_type() -> None:
    """Test that error isn't raised when getting graph with invalid input type."""

    class InvalidInputTypeRunnable(Runnable[int, int]):
        @property
        def InputType(self) -> type:
            raise TypeError()

        def invoke(
            self,
            input: int,
            config: Optional[RunnableConfig] = None,
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
        def OutputType(self) -> type:
            raise TypeError()

        def invoke(
            self,
            input: int,
            config: Optional[RunnableConfig] = None,
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
