from syrupy import SnapshotAssertion

from langchain_core.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.xml import XMLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import Runnable
from tests.unit_tests.fake.llm import FakeListLLM


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
    assert graph.draw_ascii() == snapshot


def test_graph_sequence(snapshot: SnapshotAssertion) -> None:
    fake_llm = FakeListLLM(responses=["a"])
    prompt = PromptTemplate.from_template("Hello, {name}!")
    list_parser = CommaSeparatedListOutputParser()

    sequence = prompt | fake_llm | list_parser
    graph = sequence.get_graph()
    assert graph.draw_ascii() == snapshot


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
    assert graph.draw_ascii() == snapshot
