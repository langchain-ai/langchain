from typing import Optional

from syrupy import SnapshotAssertion

from langchain_core.language_models import FakeListLLM
from langchain_core.output_parsers.list import CommaSeparatedListOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.output_parsers.xml import XMLOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.base import Runnable, RunnableConfig
from langchain_core.runnables.graph_mermaid import _escape_node_label
from tests.unit_tests.stubs import AnyStr


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
    assert graph.to_json(with_schemas=True) == {
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
                    "id": ["langchain_core", "language_models", "fake", "FakeListLLM"],
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
                        "ToolCall": {
                            "title": "ToolCall",
                            "type": "object",
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "args": {"title": "Args", "type": "object"},
                                "id": {"title": "Id", "type": "string"},
                            },
                            "required": ["name", "args", "id"],
                        },
                        "InvalidToolCall": {
                            "title": "InvalidToolCall",
                            "type": "object",
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "args": {"title": "Args", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                                "error": {"title": "Error", "type": "string"},
                            },
                            "required": ["name", "args", "id", "error"],
                        },
                        "UsageMetadata": {
                            "title": "UsageMetadata",
                            "type": "object",
                            "properties": {
                                "input_tokens": {
                                    "title": "Input Tokens",
                                    "type": "integer",
                                },
                                "output_tokens": {
                                    "title": "Output Tokens",
                                    "type": "integer",
                                },
                                "total_tokens": {
                                    "title": "Total Tokens",
                                    "type": "integer",
                                },
                            },
                            "required": [
                                "input_tokens",
                                "output_tokens",
                                "total_tokens",
                            ],
                        },
                        "AIMessage": {
                            "title": "AIMessage",
                            "description": "Message from an AI.\n\nAIMessage is returned from a chat model as a response to a prompt.\n\nThis message represents the output of the model and consists of both\nthe raw output as returned by the model together standardized fields\n(e.g., tool calls, usage metadata) added by the LangChain framework.",
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "ai",
                                    "enum": ["ai"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                                "example": {
                                    "title": "Example",
                                    "default": False,
                                    "type": "boolean",
                                },
                                "tool_calls": {
                                    "title": "Tool Calls",
                                    "default": [],
                                    "type": "array",
                                    "items": {"$ref": "#/definitions/ToolCall"},
                                },
                                "invalid_tool_calls": {
                                    "title": "Invalid Tool Calls",
                                    "default": [],
                                    "type": "array",
                                    "items": {"$ref": "#/definitions/InvalidToolCall"},
                                },
                                "usage_metadata": {
                                    "$ref": "#/definitions/UsageMetadata"
                                },
                            },
                            "required": ["content"],
                        },
                        "HumanMessage": {
                            "title": "HumanMessage",
                            "description": 'Message from a human.\n\nHumanMessages are messages that are passed in from a human to the model.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Instantiate a chat model and invoke it with the messages\n        model = ...\n        print(model.invoke(messages))',
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "human",
                                    "enum": ["human"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
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
                            "description": "Message that can be assigned an arbitrary speaker (i.e. role).",
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "chat",
                                    "enum": ["chat"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                                "role": {"title": "Role", "type": "string"},
                            },
                            "required": ["content", "role"],
                        },
                        "SystemMessage": {
                            "title": "SystemMessage",
                            "description": 'Message for priming AI behavior.\n\nThe system message is usually passed in as the first of a sequence\nof input messages.\n\nExample:\n\n    .. code-block:: python\n\n        from langchain_core.messages import HumanMessage, SystemMessage\n\n        messages = [\n            SystemMessage(\n                content="You are a helpful assistant! Your name is Bob."\n            ),\n            HumanMessage(\n                content="What is your name?"\n            )\n        ]\n\n        # Define a chat model and invoke it with the messages\n        print(model.invoke(messages))',
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "system",
                                    "enum": ["system"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                            },
                            "required": ["content"],
                        },
                        "FunctionMessage": {
                            "title": "FunctionMessage",
                            "description": "Message for passing the result of executing a tool back to a model.\n\nFunctionMessage are an older version of the ToolMessage schema, and\ndo not contain the tool_call_id field.\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.",
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "function",
                                    "enum": ["function"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                            },
                            "required": ["content", "name"],
                        },
                        "ToolMessage": {
                            "title": "ToolMessage",
                            "description": "Message for passing the result of executing a tool back to a model.\n\nToolMessages contain the result of a tool invocation. Typically, the result\nis encoded inside the `content` field.\n\nExample: A TooMessage representing a result of 42 from a tool call with id\n\n    .. code-block:: python\n\n        from langchain_core.messages import ToolMessage\n\n        ToolMessage(content='42', tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL')\n\nThe tool_call_id field is used to associate the tool call request with the\ntool call response. This is useful in situations where a chat model is able\nto request multiple tool calls in parallel.",
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
                                "response_metadata": {
                                    "title": "Response Metadata",
                                    "type": "object",
                                },
                                "type": {
                                    "title": "Type",
                                    "default": "tool",
                                    "enum": ["tool"],
                                    "type": "string",
                                },
                                "name": {"title": "Name", "type": "string"},
                                "id": {"title": "Id", "type": "string"},
                                "tool_call_id": {
                                    "title": "Tool Call Id",
                                    "type": "string",
                                },
                                "is_error": {
                                    "title": "Is Error",
                                    "default": False,
                                    "type": "boolean",
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
            },
            {
                "id": 3,
                "type": "schema",
                "data": "Parallel<as_list,as_str>Input",
            },
            {
                "id": 4,
                "type": "schema",
                "data": "Parallel<as_list,as_str>Output",
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
                "data": "conditional_str_parser_input",
            },
            {
                "id": 7,
                "type": "schema",
                "data": "conditional_str_parser_output",
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
