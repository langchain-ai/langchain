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
    assert graph.to_json() == {
        "nodes": [
            {
                "id": 0,
                "type": "schema",
                "data": {
                    "title": "PromptInput",
                    "type": "object",
                    "properties": {"name": {"title": "Name", "type": "string"}},
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
                    "id": ["tests", "unit_tests", "fake", "llm", "FakeListLLM"],
                    "name": "FakeListLLM",
                },
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
                    "title": "CommaSeparatedListOutputParserOutput",
                    "type": "array",
                    "items": {"type": "string"},
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
    assert graph.to_json() == {
        "nodes": [
            {
                "id": 0,
                "type": "schema",
                "data": {
                    "title": "PromptInput",
                    "type": "object",
                    "properties": {"name": {"title": "Name", "type": "string"}},
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
                    "id": ["tests", "unit_tests", "fake", "llm", "FakeListLLM"],
                    "name": "FakeListLLM",
                },
            },
            {
                "id": 3,
                "type": "schema",
                "data": {
                    "title": "RunnableParallel<as_list,as_str>Input",
                    "anyOf": [
                        {"type": "string"},
                        {"$ref": "#/definitions/AIMessage"},
                        {"$ref": "#/definitions/HumanMessage"},
                        {"$ref": "#/definitions/ChatMessage"},
                        {"$ref": "#/definitions/SystemMessage"},
                        {"$ref": "#/definitions/FunctionMessage"},
                        {"$ref": "#/definitions/ToolMessage"},
                    ],
                    "definitions": {
                        "AIMessage": {
                            "title": "AIMessage",
                            "description": "Message from an AI.",
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "ai",
                                    "enum": ["ai"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "example": {
                                    "title": "Example",
                                    "default": False,
                                    "type": "boolean",
                                },
                            },
                            "required": ["content"],
                        },
                        "HumanMessage": {
                            "title": "HumanMessage",
                            "description": "Message from a human.",
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "human",
                                    "enum": ["human"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "example": {
                                    "title": "Example",
                                    "default": False,
                                    "type": "boolean",
                                },
                            },
                            "required": ["content"],
                        },
                        "ChatMessage": {
                            "title": "ChatMessage",
                            "description": "Message that can be assigned an arbitrary speaker (i.e. role).",  # noqa: E501
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "chat",
                                    "enum": ["chat"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "role": {"title": "Role", "type": "string"},
                            },
                            "required": ["content", "role"],
                        },
                        "SystemMessage": {
                            "title": "SystemMessage",
                            "description": "Message for priming AI behavior, usually passed in as the first of a sequence\nof input messages.",  # noqa: E501
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "system",
                                    "enum": ["system"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                            },
                            "required": ["content"],
                        },
                        "FunctionMessage": {
                            "title": "FunctionMessage",
                            "description": "Message for passing the result of executing a function back to a model.",  # noqa: E501
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "function",
                                    "enum": ["function"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                            },
                            "required": ["content", "name"],
                        },
                        "ToolMessage": {
                            "title": "ToolMessage",
                            "description": "Message for passing the result of executing a tool back to a model.",  # noqa: E501
                            "type": "object",
                            "properties": {
                                "content": {
                                    "title": "Content",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "object"},
                                                ]
                                            },
                                        },
                                    ],
                                },
                                "additional_kwargs": {
                                    "title": "Additional Kwargs",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "tool",
                                    "enum": ["tool"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "tool_call_id": {
                                    "title": "Tool Call Id",
                                    "type": "string",
                                },
                            },
                            "required": ["content", "tool_call_id"],
                        },
                    },
                },
            },
            {
                "id": 4,
                "type": "schema",
                "data": {
                    "title": "RunnableParallel<as_list,as_str>Output",
                    "type": "object",
                    "properties": {
                        "as_list": {
                            "title": "As List",
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "as_str": {"title": "As Str"},
                    },
                },
            },
            {
                "id": 5,
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
                "id": 6,
                "type": "schema",
                "data": {"title": "conditional_str_parser_input", "type": "string"},
            },
            {
                "id": 7,
                "type": "schema",
                "data": {"title": "conditional_str_parser_output"},
            },
            {
                "id": 8,
                "type": "runnable",
                "data": {
                    "id": ["langchain", "schema", "output_parser", "StrOutputParser"],
                    "name": "StrOutputParser",
                },
            },
            {
                "id": 9,
                "type": "runnable",
                "data": {
                    "id": [
                        "langchain_core",
                        "output_parsers",
                        "xml",
                        "XMLOutputParser",
                    ],
                    "name": "XMLOutputParser",
                },
            },
        ],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 3, "target": 5},
            {"source": 5, "target": 4},
            {"source": 6, "target": 8},
            {"source": 8, "target": 7},
            {"source": 6, "target": 9},
            {"source": 9, "target": 7},
            {"source": 3, "target": 6},
            {"source": 7, "target": 4},
            {"source": 2, "target": 3},
        ],
    }
    assert graph.draw_ascii() == snapshot
