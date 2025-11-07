from typing import Any
from unittest.mock import MagicMock, patch

from packaging import version
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.xml import XMLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.graph import Edge, Graph, MermaidDrawMethod, Node
from langchain_core.runnables.graph_mermaid import (
    _render_mermaid_using_api,
    _to_safe_id,
    draw_mermaid_png,
)
from langchain_core.utils.pydantic import PYDANTIC_VERSION
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

    # trim_first_node() should not remove __start__ since it has 2 outgoing edges
    graph.trim_first_node()
    assert graph.first_node() is start

    # trim_last_node() should not remove __end__ since it has 2 incoming edges
    graph.trim_last_node()
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

    def conditional_str_parser(value: str) -> Runnable:
        if value == "a":
            return str_parser
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

    if version.parse("2.10") <= PYDANTIC_VERSION:
        assert _normalize_schema(graph.to_json(with_schemas=True)) == snapshot(
            name="graph_with_schema"
        )
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


def test_triple_nested_subgraph_mermaid(snapshot: SnapshotAssertion) -> None:
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
        "child:child_1:grandchild_1:greatgrandchild": Node(
            id="child:child_1:grandchild_1:greatgrandchild",
            name="greatgrandchild",
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
            target="child:child_1:grandchild_1:greatgrandchild",
            data=None,
            conditional=False,
        ),
        Edge(
            source="child:child_1:grandchild_1:greatgrandchild",
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


def test_single_node_subgraph_mermaid(snapshot: SnapshotAssertion) -> None:
    empty_data = BaseModel
    nodes = {
        "__start__": Node(
            id="__start__", name="__start__", data=empty_data, metadata=None
        ),
        "sub:meow": Node(id="sub:meow", name="meow", data=empty_data, metadata=None),
        "__end__": Node(id="__end__", name="__end__", data=empty_data, metadata=None),
    }
    edges = [
        Edge(source="__start__", target="sub:meow", data=None, conditional=False),
        Edge(source="sub:meow", target="__end__", data=None, conditional=False),
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
            config: RunnableConfig | None = None,
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
            config: RunnableConfig | None = None,
            **kwargs: Any,
        ) -> int:
            return input

    runnable = InvalidOutputTypeRunnable()
    # check whether runnable.invoke works
    assert runnable.invoke(1) == 1
    # check whether runnable.get_graph works
    runnable.get_graph()


def test_graph_mermaid_to_safe_id() -> None:
    """Test that node labels are correctly preprocessed for draw_mermaid."""
    assert _to_safe_id("foo") == "foo"
    assert _to_safe_id("foo-bar") == "foo-bar"
    assert _to_safe_id("foo_1") == "foo_1"
    assert _to_safe_id("#foo*&!") == "\\23foo\\2a\\26\\21"


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


def test_graph_mermaid_frontmatter_config(snapshot: SnapshotAssertion) -> None:
    graph = Graph(
        nodes={
            "__start__": Node(
                id="__start__", name="__start__", data=BaseModel, metadata=None
            ),
            "my_node": Node(
                id="my_node", name="my_node", data=BaseModel, metadata=None
            ),
        },
        edges=[
            Edge(source="__start__", target="my_node", data=None, conditional=False)
        ],
    )
    assert graph.draw_mermaid(
        frontmatter_config={
            "config": {
                "theme": "neutral",
                "look": "handDrawn",
                "themeVariables": {"primaryColor": "#e2e2e2"},
            }
        }
    ) == snapshot(name="mermaid")


def test_mermaid_base_url_default() -> None:
    """Test that _render_mermaid_using_api defaults to mermaid.ink when None."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake image data"

    with patch("requests.get", return_value=mock_response) as mock_get:
        # Call the function with base_url=None (default)
        _render_mermaid_using_api(
            "graph TD;\n    A --> B;",
            base_url=None,
        )

        # Verify that the URL was constructed with the default base URL
        assert mock_get.called
        args = mock_get.call_args[0]
        url = args[0]  # First argument to request.get is the URL
        assert url.startswith("https://mermaid.ink")


def test_mermaid_base_url_custom() -> None:
    """Test that _render_mermaid_using_api uses custom base_url when provided."""
    custom_url = "https://custom.mermaid.com"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake image data"

    with patch("requests.get", return_value=mock_response) as mock_get:
        # Call the function with custom base_url.
        _render_mermaid_using_api(
            "graph TD;\n    A --> B;",
            base_url=custom_url,
        )

        # Verify that the URL was constructed with our custom base URL
        assert mock_get.called
        args = mock_get.call_args[0]
        url = args[0]  # First argument to request.get is the URL
        assert url.startswith(custom_url)


def test_draw_mermaid_png_function_base_url() -> None:
    """Test that draw_mermaid_png function passes base_url to API renderer."""
    custom_url = "https://custom.mermaid.com"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake image data"

    with patch("requests.get", return_value=mock_response) as mock_get:
        # Call draw_mermaid_png with custom base_url
        draw_mermaid_png(
            "graph TD;\n    A --> B;",
            draw_method=MermaidDrawMethod.API,
            base_url=custom_url,
        )

        # Verify that the URL was constructed with our custom base URL
        assert mock_get.called
        args = mock_get.call_args[0]
        url = args[0]  # First argument to request.get is the URL
        assert url.startswith(custom_url)


def test_graph_draw_mermaid_png_base_url() -> None:
    """Test that Graph.draw_mermaid_png method passes base_url to renderer."""
    custom_url = "https://custom.mermaid.com"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake image data"

    with patch("requests.get", return_value=mock_response) as mock_get:
        # Create a simple graph
        graph = Graph()
        start_node = graph.add_node(BaseModel, id="start")
        end_node = graph.add_node(BaseModel, id="end")
        graph.add_edge(start_node, end_node)

        # Call draw_mermaid_png with custom base_url
        graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API, base_url=custom_url)

        # Verify that the URL was constructed with our custom base URL
        assert mock_get.called
        args = mock_get.call_args[0]
        url = args[0]  # First argument to request.get is the URL
        assert url.startswith(custom_url)


def test_graph_mermaid_special_chars(snapshot: SnapshotAssertion) -> None:
    graph = Graph(
        nodes={
            "__start__": Node(
                id="__start__", name="__start__", data=BaseModel, metadata=None
            ),
            "开始": Node(id="开始", name="开始", data=BaseModel, metadata=None),
            "结束": Node(id="结束", name="结束", data=BaseModel, metadata=None),
            "__end__": Node(
                id="__end__", name="__end__", data=BaseModel, metadata=None
            ),
        },
        edges=[
            Edge(source="__start__", target="开始", data=None, conditional=False),
            Edge(source="开始", target="结束", data=None, conditional=False),
            Edge(source="结束", target="__end__", data=None, conditional=False),
        ],
    )
    assert graph.draw_mermaid() == snapshot(name="mermaid")
