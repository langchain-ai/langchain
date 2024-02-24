import sys
from functools import partial
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)
from uuid import UUID

import pytest
from freezegun import freeze_time
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import TypedDict

from langchain_core.callbacks.manager import (
    Callbacks,
    atrace_as_chain_group,
    trace_as_chain_group,
)
from langchain_core.documents import Document
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import (
    BaseOutputParser,
    CommaSeparatedListOutputParser,
    StrOutputParser,
)
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    AddableDict,
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
    RouterRunnable,
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableConfig,
    RunnableGenerator,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnablePick,
    RunnableSequence,
    add,
    chain,
)
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.tools import BaseTool, tool
from langchain_core.tracers import (
    BaseTracer,
    ConsoleCallbackHandler,
    Run,
    RunLog,
    RunLogPatch,
)
from langchain_core.tracers.context import collect_runs
from tests.unit_tests.fake.chat_model import FakeListChatModel
from tests.unit_tests.fake.llm import FakeListLLM, FakeStreamingListLLM


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
        if run.dotted_order:
            levels = run.dotted_order.split(".")
            processed_levels = []
            for level in levels:
                timestamp, run_id = level.split("Z")
                new_run_id = self._replace_uuid(UUID(run_id))
                processed_level = f"{timestamp}Z{new_run_id}"
                processed_levels.append(processed_level)
            new_dotted_order = ".".join(processed_levels)
        else:
            new_dotted_order = None
        return run.copy(
            update={
                "id": self._replace_uuid(run.id),
                "parent_run_id": (
                    self.uuids_map[run.parent_run_id] if run.parent_run_id else None
                ),
                "child_runs": [self._copy_run(child) for child in run.child_runs],
                "execution_order": None,
                "child_execution_order": None,
                "trace_id": self._replace_uuid(run.trace_id) if run.trace_id else None,
                "dotted_order": new_dotted_order,
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


class FakeRunnableSerializable(RunnableSerializable[str, int]):
    hello: str = ""

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


def test_schemas(snapshot: SnapshotAssertion) -> None:
    fake = FakeRunnable()  # str -> int

    assert fake.input_schema.schema() == {
        "title": "FakeRunnableInput",
        "type": "string",
    }
    assert fake.output_schema.schema() == {
        "title": "FakeRunnableOutput",
        "type": "integer",
    }
    assert fake.config_schema(include=["tags", "metadata", "run_name"]).schema() == {
        "title": "FakeRunnableConfig",
        "type": "object",
        "properties": {
            "metadata": {"title": "Metadata", "type": "object"},
            "run_name": {"title": "Run Name", "type": "string"},
            "tags": {"items": {"type": "string"}, "title": "Tags", "type": "array"},
        },
    }

    fake_bound = FakeRunnable().bind(a="b")  # str -> int

    assert fake_bound.input_schema.schema() == {
        "title": "FakeRunnableInput",
        "type": "string",
    }
    assert fake_bound.output_schema.schema() == {
        "title": "FakeRunnableOutput",
        "type": "integer",
    }

    fake_w_fallbacks = FakeRunnable().with_fallbacks((fake,))  # str -> int

    assert fake_w_fallbacks.input_schema.schema() == {
        "title": "FakeRunnableInput",
        "type": "string",
    }
    assert fake_w_fallbacks.output_schema.schema() == {
        "title": "FakeRunnableOutput",
        "type": "integer",
    }

    def typed_lambda_impl(x: str) -> int:
        return len(x)

    typed_lambda = RunnableLambda(typed_lambda_impl)  # str -> int

    assert typed_lambda.input_schema.schema() == {
        "title": "typed_lambda_impl_input",
        "type": "string",
    }
    assert typed_lambda.output_schema.schema() == {
        "title": "typed_lambda_impl_output",
        "type": "integer",
    }

    async def typed_async_lambda_impl(x: str) -> int:
        return len(x)

    typed_async_lambda: Runnable = RunnableLambda(typed_async_lambda_impl)  # str -> int

    assert typed_async_lambda.input_schema.schema() == {
        "title": "typed_async_lambda_impl_input",
        "type": "string",
    }
    assert typed_async_lambda.output_schema.schema() == {
        "title": "typed_async_lambda_impl_output",
        "type": "integer",
    }

    fake_ret = FakeRetriever()  # str -> List[Document]

    assert fake_ret.input_schema.schema() == {
        "title": "FakeRetrieverInput",
        "type": "string",
    }
    assert fake_ret.output_schema.schema() == {
        "title": "FakeRetrieverOutput",
        "type": "array",
        "items": {"$ref": "#/definitions/Document"},
        "definitions": {
            "Document": {
                "title": "Document",
                "description": "Class for storing a piece of text and associated metadata.",  # noqa: E501
                "type": "object",
                "properties": {
                    "page_content": {"title": "Page Content", "type": "string"},
                    "metadata": {"title": "Metadata", "type": "object"},
                    "type": {
                        "title": "Type",
                        "enum": ["Document"],
                        "default": "Document",
                        "type": "string",
                    },
                },
                "required": ["page_content"],
            }
        },
    }

    fake_llm = FakeListLLM(responses=["a"])  # str -> List[List[str]]

    assert fake_llm.input_schema.schema() == snapshot
    assert fake_llm.output_schema.schema() == {
        "title": "FakeListLLMOutput",
        "type": "string",
    }

    fake_chat = FakeListChatModel(responses=["a"])  # str -> List[List[str]]

    assert fake_chat.input_schema.schema() == snapshot
    assert fake_chat.output_schema.schema() == snapshot

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
            ("human", "Hello, how are you?"),
        ]
    )

    assert chat_prompt.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {
            "history": {
                "title": "History",
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/definitions/AIMessage"},
                        {"$ref": "#/definitions/HumanMessage"},
                        {"$ref": "#/definitions/ChatMessage"},
                        {"$ref": "#/definitions/SystemMessage"},
                        {"$ref": "#/definitions/FunctionMessage"},
                        {"$ref": "#/definitions/ToolMessage"},
                    ]
                },
            }
        },
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                                    "anyOf": [{"type": "string"}, {"type": "object"}]
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
                    "tool_call_id": {"title": "Tool Call Id", "type": "string"},
                },
                "required": ["content", "tool_call_id"],
            },
        },
    }
    assert chat_prompt.output_schema.schema() == snapshot

    prompt = PromptTemplate.from_template("Hello, {name}!")

    assert prompt.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"name": {"title": "Name", "type": "string"}},
    }
    assert prompt.output_schema.schema() == snapshot

    prompt_mapper = PromptTemplate.from_template("Hello, {name}!").map()

    assert prompt_mapper.input_schema.schema() == {
        "definitions": {
            "PromptInput": {
                "properties": {"name": {"title": "Name", "type": "string"}},
                "title": "PromptInput",
                "type": "object",
            }
        },
        "items": {"$ref": "#/definitions/PromptInput"},
        "type": "array",
        "title": "RunnableEach<PromptTemplate>Input",
    }
    assert prompt_mapper.output_schema.schema() == snapshot

    list_parser = CommaSeparatedListOutputParser()

    assert list_parser.input_schema.schema() == snapshot
    assert list_parser.output_schema.schema() == {
        "title": "CommaSeparatedListOutputParserOutput",
        "type": "array",
        "items": {"type": "string"},
    }

    seq = prompt | fake_llm | list_parser

    assert seq.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"name": {"title": "Name", "type": "string"}},
    }
    assert seq.output_schema.schema() == {
        "type": "array",
        "items": {"type": "string"},
        "title": "CommaSeparatedListOutputParserOutput",
    }

    router: Runnable = RouterRunnable({})

    assert router.input_schema.schema() == {
        "title": "RouterRunnableInput",
        "$ref": "#/definitions/RouterInput",
        "definitions": {
            "RouterInput": {
                "title": "RouterInput",
                "type": "object",
                "properties": {
                    "key": {"title": "Key", "type": "string"},
                    "input": {"title": "Input"},
                },
                "required": ["key", "input"],
            }
        },
    }
    assert router.output_schema.schema() == {"title": "RouterRunnableOutput"}

    seq_w_map: Runnable = (
        prompt
        | fake_llm
        | {
            "original": RunnablePassthrough(input_type=str),
            "as_list": list_parser,
            "length": typed_lambda_impl,
        }
    )

    assert seq_w_map.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"name": {"title": "Name", "type": "string"}},
    }
    assert seq_w_map.output_schema.schema() == {
        "title": "RunnableParallel<original,as_list,length>Output",
        "type": "object",
        "properties": {
            "original": {"title": "Original", "type": "string"},
            "length": {"title": "Length", "type": "integer"},
            "as_list": {
                "title": "As List",
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def test_passthrough_assign_schema() -> None:
    retriever = FakeRetriever()  # str -> List[Document]
    prompt = PromptTemplate.from_template("{context} {question}")
    fake_llm = FakeListLLM(responses=["a"])  # str -> List[List[str]]

    seq_w_assign: Runnable = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | prompt
        | fake_llm
    )

    assert seq_w_assign.input_schema.schema() == {
        "properties": {"question": {"title": "Question", "type": "string"}},
        "title": "RunnableSequenceInput",
        "type": "object",
    }
    assert seq_w_assign.output_schema.schema() == {
        "title": "FakeListLLMOutput",
        "type": "string",
    }

    invalid_seq_w_assign: Runnable = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | fake_llm
    )

    # fallback to RunnableAssign.input_schema if next runnable doesn't have
    # expected dict input_schema
    assert invalid_seq_w_assign.input_schema.schema() == {
        "properties": {"question": {"title": "Question"}},
        "title": "RunnableParallel<context>Input",
        "type": "object",
    }


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_lambda_schemas() -> None:
    first_lambda = lambda x: x["hello"]  # noqa: E731
    assert RunnableLambda(first_lambda).input_schema.schema() == {
        "title": "RunnableLambdaInput",
        "type": "object",
        "properties": {"hello": {"title": "Hello"}},
    }

    second_lambda = lambda x, y: (x["hello"], x["bye"], y["bah"])  # noqa: E731
    assert RunnableLambda(second_lambda).input_schema.schema() == {  # type: ignore[arg-type]
        "title": "RunnableLambdaInput",
        "type": "object",
        "properties": {"hello": {"title": "Hello"}, "bye": {"title": "Bye"}},
    }

    def get_value(input):  # type: ignore[no-untyped-def]
        return input["variable_name"]

    assert RunnableLambda(get_value).input_schema.schema() == {
        "title": "get_value_input",
        "type": "object",
        "properties": {"variable_name": {"title": "Variable Name"}},
    }

    async def aget_value(input):  # type: ignore[no-untyped-def]
        return (input["variable_name"], input.get("another"))

    assert RunnableLambda(aget_value).input_schema.schema() == {
        "title": "aget_value_input",
        "type": "object",
        "properties": {
            "another": {"title": "Another"},
            "variable_name": {"title": "Variable Name"},
        },
    }

    async def aget_values(input):  # type: ignore[no-untyped-def]
        return {
            "hello": input["variable_name"],
            "bye": input["variable_name"],
            "byebye": input["yo"],
        }

    assert RunnableLambda(aget_values).input_schema.schema() == {
        "title": "aget_values_input",
        "type": "object",
        "properties": {
            "variable_name": {"title": "Variable Name"},
            "yo": {"title": "Yo"},
        },
    }

    class InputType(TypedDict):
        variable_name: str
        yo: int

    class OutputType(TypedDict):
        hello: str
        bye: str
        byebye: int

    async def aget_values_typed(input: InputType) -> OutputType:
        return {
            "hello": input["variable_name"],
            "bye": input["variable_name"],
            "byebye": input["yo"],
        }

    assert (
        RunnableLambda(
            aget_values_typed  # type: ignore[arg-type]
        ).input_schema.schema()
        == {
            "title": "aget_values_typed_input",
            "$ref": "#/definitions/InputType",
            "definitions": {
                "InputType": {
                    "properties": {
                        "variable_name": {
                            "title": "Variable " "Name",
                            "type": "string",
                        },
                        "yo": {"title": "Yo", "type": "integer"},
                    },
                    "required": ["variable_name", "yo"],
                    "title": "InputType",
                    "type": "object",
                }
            },
        }
    )

    assert RunnableLambda(aget_values_typed).output_schema.schema() == {  # type: ignore[arg-type]
        "title": "aget_values_typed_output",
        "$ref": "#/definitions/OutputType",
        "definitions": {
            "OutputType": {
                "properties": {
                    "bye": {"title": "Bye", "type": "string"},
                    "byebye": {"title": "Byebye", "type": "integer"},
                    "hello": {"title": "Hello", "type": "string"},
                },
                "required": ["hello", "bye", "byebye"],
                "title": "OutputType",
                "type": "object",
            }
        },
    }


def test_with_types_with_type_generics() -> None:
    """Verify that with_types works if we use things like List[int]"""

    def foo(x: int) -> None:
        """Add one to the input."""
        raise NotImplementedError()

    # Try specifying some
    RunnableLambda(foo).with_types(
        output_type=List[int],  # type: ignore[arg-type]
        input_type=List[int],  # type: ignore[arg-type]
    )
    RunnableLambda(foo).with_types(
        output_type=Sequence[int],  # type: ignore[arg-type]
        input_type=Sequence[int],  # type: ignore[arg-type]
    )


def test_schema_complex_seq() -> None:
    prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
    prompt2 = ChatPromptTemplate.from_template(
        "what country is the city {city} in? respond in {language}"
    )

    model = FakeListChatModel(responses=[""])

    chain1: Runnable = RunnableSequence(
        prompt1, model, StrOutputParser(), name="city_chain"
    )

    assert chain1.name == "city_chain"

    chain2: Runnable = (
        {"city": chain1, "language": itemgetter("language")}
        | prompt2
        | model
        | StrOutputParser()
    )

    assert chain2.input_schema.schema() == {
        "title": "RunnableParallel<city,language>Input",
        "type": "object",
        "properties": {
            "person": {"title": "Person", "type": "string"},
            "language": {"title": "Language"},
        },
    }

    assert chain2.output_schema.schema() == {
        "title": "StrOutputParserOutput",
        "type": "string",
    }

    assert chain2.with_types(input_type=str).input_schema.schema() == {
        "title": "RunnableSequenceInput",
        "type": "string",
    }

    assert chain2.with_types(input_type=int).output_schema.schema() == {
        "title": "StrOutputParserOutput",
        "type": "string",
    }

    class InputType(BaseModel):
        person: str

    assert chain2.with_types(input_type=InputType).input_schema.schema() == {
        "title": "InputType",
        "type": "object",
        "properties": {"person": {"title": "Person", "type": "string"}},
        "required": ["person"],
    }


def test_configurable_fields() -> None:
    fake_llm = FakeListLLM(responses=["a"])  # str -> List[List[str]]

    assert fake_llm.invoke("...") == "a"

    fake_llm_configurable = fake_llm.configurable_fields(
        responses=ConfigurableField(
            id="llm_responses",
            name="LLM Responses",
            description="A list of fake responses for this LLM",
        )
    )

    assert fake_llm_configurable.invoke("...") == "a"

    assert fake_llm_configurable.config_schema().schema() == {
        "title": "RunnableConfigurableFieldsConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "llm_responses": {
                        "title": "LLM Responses",
                        "description": "A list of fake responses for this LLM",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            }
        },
    }

    fake_llm_configured = fake_llm_configurable.with_config(
        configurable={"llm_responses": ["b"]}
    )

    assert fake_llm_configured.invoke("...") == "b"

    prompt = PromptTemplate.from_template("Hello, {name}!")

    assert prompt.invoke({"name": "John"}) == StringPromptValue(text="Hello, John!")

    prompt_configurable = prompt.configurable_fields(
        template=ConfigurableField(
            id="prompt_template",
            name="Prompt Template",
            description="The prompt template for this chain",
        )
    )

    assert prompt_configurable.invoke({"name": "John"}) == StringPromptValue(
        text="Hello, John!"
    )

    assert prompt_configurable.config_schema().schema() == {
        "title": "RunnableConfigurableFieldsConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "prompt_template": {
                        "title": "Prompt Template",
                        "description": "The prompt template for this chain",
                        "default": "Hello, {name}!",
                        "type": "string",
                    }
                },
            }
        },
    }

    prompt_configured = prompt_configurable.with_config(
        configurable={"prompt_template": "Hello, {name}! {name}!"}
    )

    assert prompt_configured.invoke({"name": "John"}) == StringPromptValue(
        text="Hello, John! John!"
    )

    assert prompt_configurable.with_config(
        configurable={"prompt_template": "Hello {name} in {lang}"}
    ).input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {
            "lang": {"title": "Lang", "type": "string"},
            "name": {"title": "Name", "type": "string"},
        },
    }

    chain_configurable = prompt_configurable | fake_llm_configurable | StrOutputParser()

    assert chain_configurable.invoke({"name": "John"}) == "a"

    assert chain_configurable.config_schema().schema() == {
        "title": "RunnableSequenceConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "llm_responses": {
                        "title": "LLM Responses",
                        "description": "A list of fake responses for this LLM",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "prompt_template": {
                        "title": "Prompt Template",
                        "description": "The prompt template for this chain",
                        "default": "Hello, {name}!",
                        "type": "string",
                    },
                },
            }
        },
    }

    assert (
        chain_configurable.with_config(
            configurable={
                "prompt_template": "A very good morning to you, {name} {lang}!",
                "llm_responses": ["c"],
            }
        ).invoke({"name": "John", "lang": "en"})
        == "c"
    )

    assert chain_configurable.with_config(
        configurable={
            "prompt_template": "A very good morning to you, {name} {lang}!",
            "llm_responses": ["c"],
        }
    ).input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {
            "lang": {"title": "Lang", "type": "string"},
            "name": {"title": "Name", "type": "string"},
        },
    }

    chain_with_map_configurable: Runnable = prompt_configurable | {
        "llm1": fake_llm_configurable | StrOutputParser(),
        "llm2": fake_llm_configurable | StrOutputParser(),
        "llm3": fake_llm.configurable_fields(
            responses=ConfigurableField("other_responses")
        )
        | StrOutputParser(),
    }

    assert chain_with_map_configurable.invoke({"name": "John"}) == {
        "llm1": "a",
        "llm2": "a",
        "llm3": "a",
    }

    assert chain_with_map_configurable.config_schema().schema() == {
        "title": "RunnableSequenceConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "llm_responses": {
                        "title": "LLM Responses",
                        "description": "A list of fake responses for this LLM",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "other_responses": {
                        "title": "Other Responses",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "prompt_template": {
                        "title": "Prompt Template",
                        "description": "The prompt template for this chain",
                        "default": "Hello, {name}!",
                        "type": "string",
                    },
                },
            }
        },
    }

    assert chain_with_map_configurable.with_config(
        configurable={
            "prompt_template": "A very good morning to you, {name}!",
            "llm_responses": ["c"],
            "other_responses": ["d"],
        }
    ).invoke({"name": "John"}) == {"llm1": "c", "llm2": "c", "llm3": "d"}


def test_configurable_alts_factory() -> None:
    fake_llm = FakeListLLM(responses=["a"]).configurable_alternatives(
        ConfigurableField(id="llm", name="LLM"),
        chat=partial(FakeListLLM, responses=["b"]),
    )

    assert fake_llm.invoke("...") == "a"

    assert fake_llm.with_config(configurable={"llm": "chat"}).invoke("...") == "b"


def test_configurable_fields_prefix_keys() -> None:
    fake_chat = FakeListChatModel(responses=["b"]).configurable_fields(
        responses=ConfigurableFieldMultiOption(
            id="responses",
            name="Chat Responses",
            options={
                "hello": "A good morning to you!",
                "bye": "See you later!",
                "helpful": "How can I help you?",
            },
            default=["hello", "bye"],
        ),
        # (sleep is a configurable field in FakeListChatModel)
        sleep=ConfigurableField(
            id="chat_sleep",
            is_shared=True,
        ),
    )
    fake_llm = (
        FakeListLLM(responses=["a"])
        .configurable_fields(
            responses=ConfigurableField(
                id="responses",
                name="LLM Responses",
                description="A list of fake responses for this LLM",
            )
        )
        .configurable_alternatives(
            ConfigurableField(id="llm", name="LLM"),
            chat=fake_chat | StrOutputParser(),
            prefix_keys=True,
        )
    )
    prompt = PromptTemplate.from_template("Hello, {name}!").configurable_fields(
        template=ConfigurableFieldSingleOption(
            id="prompt_template",
            name="Prompt Template",
            description="The prompt template for this chain",
            options={
                "hello": "Hello, {name}!",
                "good_morning": "A very good morning to you, {name}!",
            },
            default="hello",
        )
    )

    chain = prompt | fake_llm

    assert chain.config_schema().schema() == {
        "title": "RunnableSequenceConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "LLM": {
                "title": "LLM",
                "description": "An enumeration.",
                "enum": ["chat", "default"],
                "type": "string",
            },
            "Chat_Responses": {
                "title": "Chat Responses",
                "description": "An enumeration.",
                "enum": ["hello", "bye", "helpful"],
                "type": "string",
            },
            "Prompt_Template": {
                "title": "Prompt Template",
                "description": "An enumeration.",
                "enum": ["hello", "good_morning"],
                "type": "string",
            },
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "prompt_template": {
                        "title": "Prompt Template",
                        "description": "The prompt template for this chain",
                        "default": "hello",
                        "allOf": [{"$ref": "#/definitions/Prompt_Template"}],
                    },
                    "llm": {
                        "title": "LLM",
                        "default": "default",
                        "allOf": [{"$ref": "#/definitions/LLM"}],
                    },
                    # not prefixed because marked as shared
                    "chat_sleep": {
                        "title": "Chat Sleep",
                        "type": "number",
                    },
                    # prefixed for "chat" option
                    "llm==chat/responses": {
                        "title": "Chat Responses",
                        "default": ["hello", "bye"],
                        "type": "array",
                        "items": {"$ref": "#/definitions/Chat_Responses"},
                    },
                    # prefixed for "default" option
                    "llm==default/responses": {
                        "title": "LLM Responses",
                        "description": "A list of fake responses for this LLM",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


def test_configurable_fields_example() -> None:
    fake_chat = FakeListChatModel(responses=["b"]).configurable_fields(
        responses=ConfigurableFieldMultiOption(
            id="chat_responses",
            name="Chat Responses",
            options={
                "hello": "A good morning to you!",
                "bye": "See you later!",
                "helpful": "How can I help you?",
            },
            default=["hello", "bye"],
        )
    )
    fake_llm = (
        FakeListLLM(responses=["a"])
        .configurable_fields(
            responses=ConfigurableField(
                id="llm_responses",
                name="LLM Responses",
                description="A list of fake responses for this LLM",
            )
        )
        .configurable_alternatives(
            ConfigurableField(id="llm", name="LLM"),
            chat=fake_chat | StrOutputParser(),
        )
    )

    prompt = PromptTemplate.from_template("Hello, {name}!").configurable_fields(
        template=ConfigurableFieldSingleOption(
            id="prompt_template",
            name="Prompt Template",
            description="The prompt template for this chain",
            options={
                "hello": "Hello, {name}!",
                "good_morning": "A very good morning to you, {name}!",
            },
            default="hello",
        )
    )

    # deduplication of configurable fields
    chain_configurable = prompt | fake_llm | (lambda x: {"name": x}) | prompt | fake_llm

    assert chain_configurable.invoke({"name": "John"}) == "a"

    assert chain_configurable.config_schema().schema() == {
        "title": "RunnableSequenceConfig",
        "type": "object",
        "properties": {"configurable": {"$ref": "#/definitions/Configurable"}},
        "definitions": {
            "LLM": {
                "title": "LLM",
                "description": "An enumeration.",
                "enum": ["chat", "default"],
                "type": "string",
            },
            "Chat_Responses": {
                "description": "An enumeration.",
                "enum": ["hello", "bye", "helpful"],
                "title": "Chat Responses",
                "type": "string",
            },
            "Prompt_Template": {
                "description": "An enumeration.",
                "enum": ["hello", "good_morning"],
                "title": "Prompt Template",
                "type": "string",
            },
            "Configurable": {
                "title": "Configurable",
                "type": "object",
                "properties": {
                    "chat_responses": {
                        "default": ["hello", "bye"],
                        "items": {"$ref": "#/definitions/Chat_Responses"},
                        "title": "Chat Responses",
                        "type": "array",
                    },
                    "llm": {
                        "title": "LLM",
                        "default": "default",
                        "allOf": [{"$ref": "#/definitions/LLM"}],
                    },
                    "llm_responses": {
                        "title": "LLM Responses",
                        "description": "A list of fake responses for this LLM",
                        "default": ["a"],
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "prompt_template": {
                        "title": "Prompt Template",
                        "description": "The prompt template for this chain",
                        "default": "hello",
                        "allOf": [{"$ref": "#/definitions/Prompt_Template"}],
                    },
                },
            },
        },
    }

    with pytest.raises(ValueError):
        chain_configurable.with_config(configurable={"llm123": "chat"})

    assert (
        chain_configurable.with_config(configurable={"llm": "chat"}).invoke(
            {"name": "John"}
        )
        == "A good morning to you!"
    )

    assert (
        chain_configurable.with_config(
            configurable={"llm": "chat", "chat_responses": ["helpful"]}
        ).invoke({"name": "John"})
        == "How can I help you?"
    )


async def test_passthrough_tap_async(mocker: MockerFixture) -> None:
    fake = FakeRunnable()
    mock = mocker.Mock()

    seq: Runnable = fake | RunnablePassthrough(mock)

    assert await seq.ainvoke("hello") == 5
    assert mock.call_args_list == [mocker.call(5)]
    mock.reset_mock()

    assert [
        part async for part in seq.astream("hello", dict(metadata={"key": "value"}))
    ] == [5]
    assert mock.call_args_list == [mocker.call(5)]
    mock.reset_mock()

    assert seq.invoke("hello") == 5
    assert mock.call_args_list == [mocker.call(5)]
    mock.reset_mock()

    assert [part for part in seq.stream("hello", dict(metadata={"key": "value"}))] == [
        5
    ]
    assert mock.call_args_list == [mocker.call(5)]
    mock.reset_mock()


async def test_with_config_metadata_passthrough(mocker: MockerFixture) -> None:
    fake = FakeRunnableSerializable()
    spy = mocker.spy(fake.__class__, "invoke")
    fakew = fake.configurable_fields(hello=ConfigurableField(id="hello", name="Hello"))

    assert (
        fakew.with_config(tags=["a-tag"]).invoke(
            "hello", {"configurable": {"hello": "there"}, "metadata": {"bye": "now"}}
        )
        == 5
    )
    assert spy.call_args_list[0].args[1:] == (
        "hello",
        dict(
            tags=["a-tag"],
            callbacks=None,
            recursion_limit=25,
            configurable={"hello": "there"},
            metadata={"hello": "there", "bye": "now"},
        ),
    )
    spy.reset_mock()


async def test_with_config(mocker: MockerFixture) -> None:
    fake = FakeRunnable()
    spy = mocker.spy(fake, "invoke")

    assert fake.with_config(tags=["a-tag"]).invoke("hello") == 5
    assert spy.call_args_list == [
        mocker.call("hello", dict(tags=["a-tag"])),
    ]
    spy.reset_mock()

    fake_1: Runnable = RunnablePassthrough()
    fake_2: Runnable = RunnablePassthrough()
    spy_seq_step = mocker.spy(fake_1.__class__, "invoke")

    sequence = fake_1.with_config(tags=["a-tag"]) | fake_2.with_config(
        tags=["b-tag"], max_concurrency=5
    )
    assert sequence.invoke("hello") == "hello"
    assert len(spy_seq_step.call_args_list) == 2
    for i, call in enumerate(spy_seq_step.call_args_list):
        assert call.args[1] == "hello"
        if i == 0:
            assert call.args[2].get("tags") == ["a-tag"]
            assert call.args[2].get("max_concurrency") is None
        else:
            assert call.args[2].get("tags") == ["b-tag"]
            assert call.args[2].get("max_concurrency") == 5
    mocker.stop(spy_seq_step)

    assert [
        *fake.with_config(tags=["a-tag"]).stream(
            "hello", dict(metadata={"key": "value"})
        )
    ] == [5]
    assert spy.call_args_list == [
        mocker.call("hello", dict(tags=["a-tag"], metadata={"key": "value"})),
    ]
    spy.reset_mock()

    assert fake.with_config(recursion_limit=5).batch(
        ["hello", "wooorld"], [dict(tags=["a-tag"]), dict(metadata={"key": "value"})]
    ) == [5, 7]

    assert len(spy.call_args_list) == 2
    for i, call in enumerate(
        sorted(spy.call_args_list, key=lambda x: 0 if x.args[0] == "hello" else 1)
    ):
        assert call.args[0] == ("hello" if i == 0 else "wooorld")
        if i == 0:
            assert call.args[1].get("recursion_limit") == 5
            assert call.args[1].get("tags") == ["a-tag"]
            assert call.args[1].get("metadata") == {}
        else:
            assert call.args[1].get("recursion_limit") == 5
            assert call.args[1].get("tags") == []
            assert call.args[1].get("metadata") == {"key": "value"}

    spy.reset_mock()

    assert fake.with_config(metadata={"a": "b"}).batch(
        ["hello", "wooorld"], dict(tags=["a-tag"])
    ) == [5, 7]
    assert len(spy.call_args_list) == 2
    for i, call in enumerate(spy.call_args_list):
        assert call.args[0] == ("hello" if i == 0 else "wooorld")
        assert call.args[1].get("tags") == ["a-tag"]
        assert call.args[1].get("metadata") == {"a": "b"}
    spy.reset_mock()

    handler = ConsoleCallbackHandler()
    assert (
        await fake.with_config(metadata={"a": "b"}).ainvoke(
            "hello", config={"callbacks": [handler]}
        )
        == 5
    )
    assert spy.call_args_list == [
        mocker.call("hello", dict(callbacks=[handler], metadata={"a": "b"})),
    ]
    spy.reset_mock()

    assert [
        part async for part in fake.with_config(metadata={"a": "b"}).astream("hello")
    ] == [5]
    assert spy.call_args_list == [
        mocker.call("hello", dict(metadata={"a": "b"})),
    ]
    spy.reset_mock()

    assert await fake.with_config(recursion_limit=5, tags=["c"]).abatch(
        ["hello", "wooorld"], dict(metadata={"key": "value"})
    ) == [
        5,
        7,
    ]
    assert spy.call_args_list == [
        mocker.call(
            "hello",
            dict(
                metadata={"key": "value"},
                tags=["c"],
                callbacks=None,
                recursion_limit=5,
            ),
        ),
        mocker.call(
            "wooorld",
            dict(
                metadata={"key": "value"},
                tags=["c"],
                callbacks=None,
                recursion_limit=5,
            ),
        ),
    ]


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

    assert len(spy.call_args_list) == 2
    for i, call in enumerate(spy.call_args_list):
        assert call.args[0] == ("hello" if i == 0 else "wooorld")
        if i == 0:
            assert call.args[1].get("tags") == ["a-tag"]
            assert call.args[1].get("metadata") == {}
        else:
            assert call.args[1].get("tags") == []
            assert call.args[1].get("metadata") == {"key": "value"}

    spy.reset_mock()

    assert fake.batch(["hello", "wooorld"], dict(tags=["a-tag"])) == [5, 7]
    assert len(spy.call_args_list) == 2
    for i, call in enumerate(spy.call_args_list):
        assert call.args[0] == ("hello" if i == 0 else "wooorld")
        assert call.args[1].get("tags") == ["a-tag"]
        assert call.args[1].get("metadata") == {}
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
        mocker.call(
            "hello",
            dict(
                metadata={"key": "value"},
                tags=[],
                callbacks=None,
                recursion_limit=25,
            ),
        ),
        mocker.call(
            "wooorld",
            dict(
                metadata={"key": "value"},
                tags=[],
                callbacks=None,
                recursion_limit=25,
            ),
        ),
    ]


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

    stream_log = [
        part async for part in prompt.astream_log({"question": "What is your name?"})
    ]

    assert len(stream_log[0].ops) == 1
    assert stream_log[0].ops[0]["op"] == "replace"
    assert stream_log[0].ops[0]["path"] == ""
    assert stream_log[0].ops[0]["value"]["logs"] == {}
    assert stream_log[0].ops[0]["value"]["final_output"] is None
    assert stream_log[0].ops[0]["value"]["streamed_output"] == []
    assert isinstance(stream_log[0].ops[0]["value"]["id"], str)

    assert stream_log[1:] == [
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": expected},
            {
                "op": "replace",
                "path": "/final_output",
                "value": ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are a nice assistant."),
                        HumanMessage(content="What is your name?"),
                    ]
                ),
            },
        ),
    ]

    stream_log_state = [
        part
        async for part in prompt.astream_log(
            {"question": "What is your name?"}, diff=False
        )
    ]

    # remove random id
    stream_log[0].ops[0]["value"]["id"] = "00000000-0000-0000-0000-000000000000"
    stream_log_state[-1].ops[0]["value"]["id"] = "00000000-0000-0000-0000-000000000000"
    stream_log_state[-1].state["id"] = "00000000-0000-0000-0000-000000000000"

    # assert output with diff=False matches output with diff=True
    assert stream_log_state[-1].ops == [op for chunk in stream_log for op in chunk.ops]
    assert stream_log_state[-1] == RunLog(
        *[op for chunk in stream_log for op in chunk.ops],
        state={
            "final_output": ChatPromptValue(
                messages=[
                    SystemMessage(content="You are a nice assistant."),
                    HumanMessage(content="What is your name?"),
                ]
            ),
            "id": "00000000-0000-0000-0000-000000000000",
            "logs": {},
            "streamed_output": [
                ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are a nice assistant."),
                        HumanMessage(content="What is your name?"),
                    ]
                )
            ],
            "type": "prompt",
            "name": "ChatPromptTemplate",
        },
    )

    # nested inside trace_with_chain_group

    async with atrace_as_chain_group("a_group") as manager:
        stream_log_nested = [
            part
            async for part in prompt.astream_log(
                {"question": "What is your name?"}, config={"callbacks": manager}
            )
        ]

    assert len(stream_log_nested[0].ops) == 1
    assert stream_log_nested[0].ops[0]["op"] == "replace"
    assert stream_log_nested[0].ops[0]["path"] == ""
    assert stream_log_nested[0].ops[0]["value"]["logs"] == {}
    assert stream_log_nested[0].ops[0]["value"]["final_output"] is None
    assert stream_log_nested[0].ops[0]["value"]["streamed_output"] == []
    assert isinstance(stream_log_nested[0].ops[0]["value"]["id"], str)

    assert stream_log_nested[1:] == [
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": expected},
            {
                "op": "replace",
                "path": "/final_output",
                "value": ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are a nice assistant."),
                        HumanMessage(content="What is your name?"),
                    ]
                ),
            },
        ),
    ]


def test_prompt_template_params() -> None:
    prompt = ChatPromptTemplate.from_template(
        "Respond to the following question: {question}"
    )
    result = prompt.invoke(
        {
            "question": "test",
            "topic": "test",
        }
    )
    assert result == ChatPromptValue(
        messages=[HumanMessage(content="Respond to the following question: test")]
    )

    with pytest.raises(KeyError):
        prompt.invoke({})


def test_with_listeners(mocker: MockerFixture) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo"])

    chain: Runnable = prompt | chat

    mock_start = mocker.Mock()
    mock_end = mocker.Mock()

    chain.with_listeners(on_start=mock_start, on_end=mock_end).invoke(
        {"question": "Who are you?"}
    )

    assert mock_start.call_count == 1
    assert mock_start.call_args[0][0].name == "RunnableSequence"
    assert mock_end.call_count == 1

    mock_start.reset_mock()
    mock_end.reset_mock()

    with trace_as_chain_group("hello") as manager:
        chain.with_listeners(on_start=mock_start, on_end=mock_end).invoke(
            {"question": "Who are you?"}, {"callbacks": manager}
        )

    assert mock_start.call_count == 1
    assert mock_start.call_args[0][0].name == "RunnableSequence"
    assert mock_end.call_count == 1


async def test_with_listeners_async(mocker: MockerFixture) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo"])

    chain: Runnable = prompt | chat

    mock_start = mocker.Mock()
    mock_end = mocker.Mock()

    await chain.with_listeners(on_start=mock_start, on_end=mock_end).ainvoke(
        {"question": "Who are you?"}
    )

    assert mock_start.call_count == 1
    assert mock_start.call_args[0][0].name == "RunnableSequence"
    assert mock_end.call_count == 1

    mock_start.reset_mock()
    mock_end.reset_mock()

    async with atrace_as_chain_group("hello") as manager:
        await chain.with_listeners(on_start=mock_start, on_end=mock_end).ainvoke(
            {"question": "Who are you?"}, {"callbacks": manager}
        )

    assert mock_start.call_count == 1
    assert mock_start.call_args[0][0].name == "RunnableSequence"
    assert mock_end.call_count == 1


@freeze_time("2023-01-01")
def test_prompt_with_chat_model(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo"])

    chain: Runnable = prompt | chat

    assert repr(chain) == snapshot
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
    ] == [
        AIMessageChunk(content="f"),
        AIMessageChunk(content="o"),
        AIMessageChunk(content="o"),
    ]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )


@freeze_time("2023-01-01")
async def test_prompt_with_chat_model_async(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    chat = FakeListChatModel(responses=["foo"])

    chain: Runnable = prompt | chat

    assert repr(chain) == snapshot
    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == []
    assert chain.last == chat
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    chat_spy = mocker.spy(chat.__class__, "ainvoke")
    tracer = FakeTracer()
    assert await chain.ainvoke(
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
    prompt_spy = mocker.spy(prompt.__class__, "abatch")
    chat_spy = mocker.spy(chat.__class__, "abatch")
    tracer = FakeTracer()
    assert await chain.abatch(
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
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    chat_spy = mocker.spy(chat.__class__, "astream")
    tracer = FakeTracer()
    assert [
        a
        async for a in chain.astream(
            {"question": "What is your name?"}, dict(callbacks=[tracer])
        )
    ] == [
        AIMessageChunk(content="f"),
        AIMessageChunk(content="o"),
        AIMessageChunk(content="o"),
    ]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert chat_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )


@freeze_time("2023-01-01")
async def test_prompt_with_llm(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeListLLM(responses=["foo", "bar"])

    chain: Runnable = prompt | llm

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

    prompt_spy.reset_mock()
    llm_spy.reset_mock()
    stream_log = [
        part async for part in chain.astream_log({"question": "What is your name?"})
    ]

    # remove ids from logs
    for part in stream_log:
        for op in part.ops:
            if (
                isinstance(op["value"], dict)
                and "id" in op["value"]
                and not isinstance(op["value"]["id"], list)  # serialized lc id
            ):
                del op["value"]["id"]

    expected = [
        RunLogPatch(
            {
                "op": "replace",
                "path": "",
                "value": {
                    "logs": {},
                    "final_output": None,
                    "streamed_output": [],
                    "name": "RunnableSequence",
                    "type": "chain",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate",
                "value": {
                    "end_time": None,
                    "final_output": None,
                    "metadata": {},
                    "name": "ChatPromptTemplate",
                    "start_time": "2023-01-01T00:00:00.000+00:00",
                    "streamed_output": [],
                    "streamed_output_str": [],
                    "tags": ["seq:step:1"],
                    "type": "prompt",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate/final_output",
                "value": ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are a nice assistant."),
                        HumanMessage(content="What is your name?"),
                    ]
                ),
            },
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate/end_time",
                "value": "2023-01-01T00:00:00.000+00:00",
            },
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/FakeListLLM",
                "value": {
                    "end_time": None,
                    "final_output": None,
                    "metadata": {},
                    "name": "FakeListLLM",
                    "start_time": "2023-01-01T00:00:00.000+00:00",
                    "streamed_output": [],
                    "streamed_output_str": [],
                    "tags": ["seq:step:2"],
                    "type": "llm",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/FakeListLLM/final_output",
                "value": {
                    "generations": [
                        [{"generation_info": None, "text": "foo", "type": "Generation"}]
                    ],
                    "llm_output": None,
                    "run": None,
                },
            },
            {
                "op": "add",
                "path": "/logs/FakeListLLM/end_time",
                "value": "2023-01-01T00:00:00.000+00:00",
            },
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": "foo"},
            {"op": "replace", "path": "/final_output", "value": "foo"},
        ),
    ]
    assert stream_log == expected


@freeze_time("2023-01-01")
async def test_prompt_with_llm_parser(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["bear, dog, cat", "tomato, lettuce, onion"])
    parser = CommaSeparatedListOutputParser()

    chain: Runnable = prompt | llm | parser

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [llm]
    assert chain.last == parser
    assert dumps(chain, pretty=True) == snapshot

    # Test invoke
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    llm_spy = mocker.spy(llm.__class__, "ainvoke")
    parser_spy = mocker.spy(parser.__class__, "ainvoke")
    tracer = FakeTracer()
    assert await chain.ainvoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == ["bear", "dog", "cat"]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert parser_spy.call_args.args[1] == "bear, dog, cat"
    assert tracer.runs == snapshot
    mocker.stop(prompt_spy)
    mocker.stop(llm_spy)
    mocker.stop(parser_spy)

    # Test batch
    prompt_spy = mocker.spy(prompt.__class__, "abatch")
    llm_spy = mocker.spy(llm.__class__, "abatch")
    parser_spy = mocker.spy(parser.__class__, "abatch")
    tracer = FakeTracer()
    assert await chain.abatch(
        [
            {"question": "What is your name?"},
            {"question": "What is your favorite color?"},
        ],
        dict(callbacks=[tracer]),
    ) == [["tomato", "lettuce", "onion"], ["bear", "dog", "cat"]]
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
    assert parser_spy.call_args.args[1] == [
        "tomato, lettuce, onion",
        "bear, dog, cat",
    ]
    assert tracer.runs == snapshot
    mocker.stop(prompt_spy)
    mocker.stop(llm_spy)
    mocker.stop(parser_spy)

    # Test stream
    prompt_spy = mocker.spy(prompt.__class__, "ainvoke")
    llm_spy = mocker.spy(llm.__class__, "astream")
    tracer = FakeTracer()
    assert [
        token
        async for token in chain.astream(
            {"question": "What is your name?"}, dict(callbacks=[tracer])
        )
    ] == [["tomato"], ["lettuce"], ["onion"]]
    assert prompt_spy.call_args.args[1] == {"question": "What is your name?"}
    assert llm_spy.call_args.args[1] == ChatPromptValue(
        messages=[
            SystemMessage(content="You are a nice assistant."),
            HumanMessage(content="What is your name?"),
        ]
    )

    prompt_spy.reset_mock()
    llm_spy.reset_mock()
    stream_log = [
        part async for part in chain.astream_log({"question": "What is your name?"})
    ]

    # remove ids from logs
    for part in stream_log:
        for op in part.ops:
            if (
                isinstance(op["value"], dict)
                and "id" in op["value"]
                and not isinstance(op["value"]["id"], list)  # serialized lc id
            ):
                del op["value"]["id"]

    expected = [
        RunLogPatch(
            {
                "op": "replace",
                "path": "",
                "value": {
                    "logs": {},
                    "final_output": None,
                    "streamed_output": [],
                    "name": "RunnableSequence",
                    "type": "chain",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate",
                "value": {
                    "end_time": None,
                    "final_output": None,
                    "metadata": {},
                    "name": "ChatPromptTemplate",
                    "start_time": "2023-01-01T00:00:00.000+00:00",
                    "streamed_output": [],
                    "streamed_output_str": [],
                    "tags": ["seq:step:1"],
                    "type": "prompt",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate/final_output",
                "value": ChatPromptValue(
                    messages=[
                        SystemMessage(content="You are a nice assistant."),
                        HumanMessage(content="What is your name?"),
                    ]
                ),
            },
            {
                "op": "add",
                "path": "/logs/ChatPromptTemplate/end_time",
                "value": "2023-01-01T00:00:00.000+00:00",
            },
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/FakeStreamingListLLM",
                "value": {
                    "end_time": None,
                    "final_output": None,
                    "metadata": {},
                    "name": "FakeStreamingListLLM",
                    "start_time": "2023-01-01T00:00:00.000+00:00",
                    "streamed_output": [],
                    "streamed_output_str": [],
                    "tags": ["seq:step:2"],
                    "type": "llm",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/FakeStreamingListLLM/final_output",
                "value": {
                    "generations": [
                        [
                            {
                                "generation_info": None,
                                "text": "bear, dog, cat",
                                "type": "Generation",
                            }
                        ]
                    ],
                    "llm_output": None,
                    "run": None,
                },
            },
            {
                "op": "add",
                "path": "/logs/FakeStreamingListLLM/end_time",
                "value": "2023-01-01T00:00:00.000+00:00",
            },
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser",
                "value": {
                    "end_time": None,
                    "final_output": None,
                    "metadata": {},
                    "name": "CommaSeparatedListOutputParser",
                    "start_time": "2023-01-01T00:00:00.000+00:00",
                    "streamed_output": [],
                    "streamed_output_str": [],
                    "tags": ["seq:step:3"],
                    "type": "parser",
                },
            }
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser/streamed_output/-",
                "value": ["bear"],
            }
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": ["bear"]},
            {"op": "replace", "path": "/final_output", "value": ["bear"]},
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser/streamed_output/-",
                "value": ["dog"],
            }
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": ["dog"]},
            {"op": "add", "path": "/final_output/1", "value": "dog"},
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser/streamed_output/-",
                "value": ["cat"],
            }
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": ["cat"]},
            {"op": "add", "path": "/final_output/2", "value": "cat"},
        ),
        RunLogPatch(
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser/final_output",
                "value": {"output": ["bear", "dog", "cat"]},
            },
            {
                "op": "add",
                "path": "/logs/CommaSeparatedListOutputParser/end_time",
                "value": "2023-01-01T00:00:00.000+00:00",
            },
        ),
    ]
    assert stream_log == expected


@freeze_time("2023-01-01")
async def test_stream_log_retriever() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{documents}"
        + "{question}"
    )
    llm = FakeListLLM(responses=["foo", "bar"])

    chain: Runnable = (
        {"documents": FakeRetriever(), "question": itemgetter("question")}
        | prompt
        | {"one": llm, "two": llm}
    )

    stream_log = [
        part async for part in chain.astream_log({"question": "What is your name?"})
    ]

    # remove ids from logs
    for part in stream_log:
        for op in part.ops:
            if (
                isinstance(op["value"], dict)
                and "id" in op["value"]
                and not isinstance(op["value"]["id"], list)  # serialized lc id
            ):
                del op["value"]["id"]

    assert sorted(cast(RunLog, add(stream_log)).state["logs"]) == [
        "ChatPromptTemplate",
        "FakeListLLM",
        "FakeListLLM:2",
        "Retriever",
        "RunnableLambda",
        "RunnableParallel<documents,question>",
        "RunnableParallel<one,two>",
    ]


@freeze_time("2023-01-01")
async def test_stream_log_lists() -> None:
    async def list_producer(input: AsyncIterator[Any]) -> AsyncIterator[AddableDict]:
        for i in range(4):
            yield AddableDict(alist=[str(i)])

    chain: Runnable = RunnableGenerator(list_producer)

    stream_log = [
        part async for part in chain.astream_log({"question": "What is your name?"})
    ]

    # remove ids from logs
    for part in stream_log:
        for op in part.ops:
            if (
                isinstance(op["value"], dict)
                and "id" in op["value"]
                and not isinstance(op["value"]["id"], list)  # serialized lc id
            ):
                del op["value"]["id"]

    assert stream_log == [
        RunLogPatch(
            {
                "op": "replace",
                "path": "",
                "value": {
                    "final_output": None,
                    "logs": {},
                    "streamed_output": [],
                    "name": "list_producer",
                    "type": "chain",
                },
            }
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": {"alist": ["0"]}},
            {"op": "replace", "path": "/final_output", "value": {"alist": ["0"]}},
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": {"alist": ["1"]}},
            {"op": "add", "path": "/final_output/alist/1", "value": "1"},
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": {"alist": ["2"]}},
            {"op": "add", "path": "/final_output/alist/2", "value": "2"},
        ),
        RunLogPatch(
            {"op": "add", "path": "/streamed_output/-", "value": {"alist": ["3"]}},
            {"op": "add", "path": "/final_output/alist/3", "value": "3"},
        ),
    ]

    state = add(stream_log)

    assert isinstance(state, RunLog)

    assert state.state == {
        "final_output": {"alist": ["0", "1", "2", "3"]},
        "logs": {},
        "name": "list_producer",
        "streamed_output": [
            {"alist": ["0"]},
            {"alist": ["1"]},
            {"alist": ["2"]},
            {"alist": ["3"]},
        ],
        "type": "chain",
    }


@freeze_time("2023-01-01")
async def test_prompt_with_llm_and_async_lambda(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeListLLM(responses=["foo", "bar"])

    async def passthrough(input: Any) -> Any:
        return input

    chain = prompt | llm | passthrough

    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [llm]
    assert chain.last == RunnableLambda(func=passthrough)
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
    if sys.version_info >= (3, 9):
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

    chain2 = cast(RunnableSequence, input_formatter | prompt2 | chat2 | parser2)

    assert isinstance(chain, RunnableSequence)
    assert chain2.first == input_formatter
    assert chain2.middle == [prompt2, chat2]
    assert chain2.last == parser2
    if sys.version_info >= (3, 9):
        assert dumps(chain2, pretty=True) == snapshot

    combined_chain = cast(RunnableSequence, chain | chain2)

    assert combined_chain.first == prompt
    assert combined_chain.middle == [
        chat,
        parser,
        input_formatter,
        prompt2,
        chat2,
    ]
    assert combined_chain.last == parser2
    if sys.version_info >= (3, 9):
        assert dumps(combined_chain, pretty=True) == snapshot

    # Test invoke
    tracer = FakeTracer()
    assert combined_chain.invoke(
        {"question": "What is your name?"}, dict(callbacks=[tracer])
    ) == ["baz", "qux"]

    if sys.version_info >= (3, 9):
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

    chain: Runnable = (
        {
            "question": RunnablePassthrough[str]() | passthrough,
            "documents": passthrough | retriever,
            "just_to_test_lambda": passthrough,
        }
        | prompt
        | chat
        | parser
    )

    assert repr(chain) == snapshot
    assert isinstance(chain, RunnableSequence)
    assert isinstance(chain.first, RunnableParallel)
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
[Document(page_content='foo'), Document(page_content='bar')]

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
    assert map_run.name == "RunnableParallel<question,documents,just_to_test_lambda>"
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

    assert repr(chain) == snapshot
    assert isinstance(chain, RunnableSequence)
    assert chain.first == prompt
    assert chain.middle == [RunnableLambda(passthrough)]
    assert isinstance(chain.last, RunnableParallel)
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
    assert map_run.name == "RunnableParallel<chat,llm>"
    assert len(map_run.child_runs) == 2


@freeze_time("2023-01-01")
async def test_router_runnable(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    chain1: Runnable = ChatPromptTemplate.from_template(
        "You are a math genius. Answer the question: {question}"
    ) | FakeListLLM(responses=["4"])
    chain2: Runnable = ChatPromptTemplate.from_template(
        "You are an english major. Answer the question: {question}"
    ) | FakeListLLM(responses=["2"])
    router: Runnable = RouterRunnable({"math": chain1, "english": chain2})
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
async def test_higher_order_lambda_runnable(
    mocker: MockerFixture, snapshot: SnapshotAssertion
) -> None:
    math_chain: Runnable = ChatPromptTemplate.from_template(
        "You are a math genius. Answer the question: {question}"
    ) | FakeListLLM(responses=["4"])
    english_chain: Runnable = ChatPromptTemplate.from_template(
        "You are an english major. Answer the question: {question}"
    ) | FakeListLLM(responses=["2"])
    input_map: Runnable = RunnableParallel(
        key=lambda x: x["key"],
        input={"question": lambda x: x["question"]},
    )

    def router(input: Dict[str, Any]) -> Runnable:
        if input["key"] == "math":
            return itemgetter("input") | math_chain
        elif input["key"] == "english":
            return itemgetter("input") | english_chain
        else:
            raise ValueError(f"Unknown key: {input['key']}")

    chain: Runnable = input_map | router
    if sys.version_info >= (3, 9):
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
    math_spy = mocker.spy(math_chain.__class__, "invoke")
    tracer = FakeTracer()
    assert (
        chain.invoke({"key": "math", "question": "2 + 2"}, dict(callbacks=[tracer]))
        == "4"
    )
    assert math_spy.call_args.args[1] == {
        "key": "math",
        "input": {"question": "2 + 2"},
    }
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 2
    router_run = parent_run.child_runs[1]
    assert router_run.name == "router"
    assert len(router_run.child_runs) == 1
    math_run = router_run.child_runs[0]
    assert math_run.name == "RunnableSequence"
    assert len(math_run.child_runs) == 3

    # Test ainvoke
    async def arouter(input: Dict[str, Any]) -> Runnable:
        if input["key"] == "math":
            return itemgetter("input") | math_chain
        elif input["key"] == "english":
            return itemgetter("input") | english_chain
        else:
            raise ValueError(f"Unknown key: {input['key']}")

    achain: Runnable = input_map | arouter
    math_spy = mocker.spy(math_chain.__class__, "ainvoke")
    tracer = FakeTracer()
    assert (
        await achain.ainvoke(
            {"key": "math", "question": "2 + 2"}, dict(callbacks=[tracer])
        )
        == "4"
    )
    assert math_spy.call_args.args[1] == {
        "key": "math",
        "input": {"question": "2 + 2"},
    }
    assert len([r for r in tracer.runs if r.parent_run_id is None]) == 1
    parent_run = next(r for r in tracer.runs if r.parent_run_id is None)
    assert len(parent_run.child_runs) == 2
    router_run = parent_run.child_runs[1]
    assert router_run.name == "arouter"
    assert len(router_run.child_runs) == 1
    math_run = router_run.child_runs[0]
    assert math_run.name == "RunnableSequence"
    assert len(math_run.child_runs) == 3


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
    assert isinstance(chain.last, RunnableParallel)
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
    assert map_run.name == "RunnableParallel<chat,llm,passthrough>"
    assert len(map_run.child_runs) == 3


def test_map_stream() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat_res = "i'm a chatbot"
    # sleep to better simulate a real stream
    chat = FakeListChatModel(responses=[chat_res], sleep=0.01)

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    chain: Runnable = prompt | {
        "chat": chat.bind(stop=["Thought:"]),
        "llm": llm,
        "passthrough": RunnablePassthrough(),
    }

    stream = chain.stream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] in [
        {"passthrough": prompt.invoke({"question": "What is your name?"})},
        {"llm": "i"},
        {"chat": AIMessageChunk(content="i")},
    ]
    assert len(streamed_chunks) == len(chat_res) + len(llm_res) + 1
    assert all(len(c.keys()) == 1 for c in streamed_chunks)
    assert final_value is not None
    assert final_value.get("chat").content == "i'm a chatbot"
    assert final_value.get("llm") == "i'm a textbot"
    assert final_value.get("passthrough") == prompt.invoke(
        {"question": "What is your name?"}
    )

    chain_pick_one = chain.pick("llm")

    assert chain_pick_one.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "string",
    }

    stream = chain_pick_one.stream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] == "i"
    assert len(streamed_chunks) == len(llm_res)

    chain_pick_two = chain.assign(hello=RunnablePick("llm").pipe(llm)).pick(
        ["llm", "hello"]
    )

    assert chain_pick_two.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "object",
        "properties": {
            "hello": {"title": "Hello", "type": "string"},
            "llm": {"title": "Llm", "type": "string"},
        },
    }

    stream = chain_pick_two.stream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] in [
        {"llm": "i"},
        {"chat": AIMessageChunk(content="i")},
    ]
    assert len(streamed_chunks) == len(llm_res) + len(chat_res)


def test_map_stream_iterator_input() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat_res = "i'm a chatbot"
    # sleep to better simulate a real stream
    chat = FakeListChatModel(responses=[chat_res], sleep=0.01)

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    chain: Runnable = (
        prompt
        | llm
        | {
            "chat": chat.bind(stop=["Thought:"]),
            "llm": llm,
            "passthrough": RunnablePassthrough(),
        }
    )

    stream = chain.stream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] in [
        {"passthrough": "i"},
        {"llm": "i"},
        {"chat": AIMessageChunk(content="i")},
    ]
    assert len(streamed_chunks) == len(chat_res) + len(llm_res) + len(llm_res)
    assert all(len(c.keys()) == 1 for c in streamed_chunks)
    assert final_value is not None
    assert final_value.get("chat").content == "i'm a chatbot"
    assert final_value.get("llm") == "i'm a textbot"
    assert final_value.get("passthrough") == "i'm a textbot"


async def test_map_astream() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat_res = "i'm a chatbot"
    # sleep to better simulate a real stream
    chat = FakeListChatModel(responses=[chat_res], sleep=0.01)

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    chain: Runnable = prompt | {
        "chat": chat.bind(stop=["Thought:"]),
        "llm": llm,
        "passthrough": RunnablePassthrough(),
    }

    stream = chain.astream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    async for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] in [
        {"passthrough": prompt.invoke({"question": "What is your name?"})},
        {"llm": "i"},
        {"chat": AIMessageChunk(content="i")},
    ]
    assert len(streamed_chunks) == len(chat_res) + len(llm_res) + 1
    assert all(len(c.keys()) == 1 for c in streamed_chunks)
    assert final_value is not None
    assert final_value.get("chat").content == "i'm a chatbot"
    assert final_value.get("llm") == "i'm a textbot"
    assert final_value.get("passthrough") == prompt.invoke(
        {"question": "What is your name?"}
    )

    # Test astream_log state accumulation

    final_state = None
    streamed_ops = []
    async for chunk in chain.astream_log({"question": "What is your name?"}):
        streamed_ops.extend(chunk.ops)
        if final_state is None:
            final_state = chunk
        else:
            final_state += chunk
    final_state = cast(RunLog, final_state)

    assert final_state.state["final_output"] == final_value
    assert len(final_state.state["streamed_output"]) == len(streamed_chunks)
    assert isinstance(final_state.state["id"], str)
    assert len(final_state.ops) == len(streamed_ops)
    assert len(final_state.state["logs"]) == 5
    assert (
        final_state.state["logs"]["ChatPromptTemplate"]["name"] == "ChatPromptTemplate"
    )
    assert final_state.state["logs"]["ChatPromptTemplate"][
        "final_output"
    ] == prompt.invoke({"question": "What is your name?"})
    assert (
        final_state.state["logs"]["RunnableParallel<chat,llm,passthrough>"]["name"]
        == "RunnableParallel<chat,llm,passthrough>"
    )
    assert sorted(final_state.state["logs"]) == [
        "ChatPromptTemplate",
        "FakeListChatModel",
        "FakeStreamingListLLM",
        "RunnableParallel<chat,llm,passthrough>",
        "RunnablePassthrough",
    ]

    # Test astream_log with include filters
    final_state = None
    async for chunk in chain.astream_log(
        {"question": "What is your name?"}, include_names=["FakeListChatModel"]
    ):
        if final_state is None:
            final_state = chunk
        else:
            final_state += chunk
    final_state = cast(RunLog, final_state)

    assert final_state.state["final_output"] == final_value
    assert len(final_state.state["streamed_output"]) == len(streamed_chunks)
    assert len(final_state.state["logs"]) == 1
    assert final_state.state["logs"]["FakeListChatModel"]["name"] == "FakeListChatModel"

    # Test astream_log with exclude filters
    final_state = None
    async for chunk in chain.astream_log(
        {"question": "What is your name?"}, exclude_names=["FakeListChatModel"]
    ):
        if final_state is None:
            final_state = chunk
        else:
            final_state += chunk
    final_state = cast(RunLog, final_state)

    assert final_state.state["final_output"] == final_value
    assert len(final_state.state["streamed_output"]) == len(streamed_chunks)
    assert len(final_state.state["logs"]) == 4
    assert (
        final_state.state["logs"]["ChatPromptTemplate"]["name"] == "ChatPromptTemplate"
    )
    assert final_state.state["logs"]["ChatPromptTemplate"]["final_output"] == (
        prompt.invoke({"question": "What is your name?"})
    )
    assert (
        final_state.state["logs"]["RunnableParallel<chat,llm,passthrough>"]["name"]
        == "RunnableParallel<chat,llm,passthrough>"
    )
    assert sorted(final_state.state["logs"]) == [
        "ChatPromptTemplate",
        "FakeStreamingListLLM",
        "RunnableParallel<chat,llm,passthrough>",
        "RunnablePassthrough",
    ]


async def test_map_astream_iterator_input() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )

    chat_res = "i'm a chatbot"
    # sleep to better simulate a real stream
    chat = FakeListChatModel(responses=[chat_res], sleep=0.01)

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    chain: Runnable = (
        prompt
        | llm
        | {
            "chat": chat.bind(stop=["Thought:"]),
            "llm": llm,
            "passthrough": RunnablePassthrough(),
        }
    )

    stream = chain.astream({"question": "What is your name?"})

    final_value = None
    streamed_chunks = []
    async for chunk in stream:
        streamed_chunks.append(chunk)
        if final_value is None:
            final_value = chunk
        else:
            final_value += chunk

    assert streamed_chunks[0] in [
        {"passthrough": "i"},
        {"llm": "i"},
        {"chat": AIMessageChunk(content="i")},
    ]
    assert len(streamed_chunks) == len(chat_res) + len(llm_res) + len(llm_res)
    assert all(len(c.keys()) == 1 for c in streamed_chunks)
    assert final_value is not None
    assert final_value.get("chat").content == "i'm a chatbot"
    assert final_value.get("llm") == "i'm a textbot"
    assert final_value.get("passthrough") == llm_res


def test_with_config_with_config() -> None:
    llm = FakeListLLM(responses=["i'm a textbot"])

    assert dumpd(
        llm.with_config({"metadata": {"a": "b"}}).with_config(tags=["a-tag"])
    ) == dumpd(llm.with_config({"metadata": {"a": "b"}, "tags": ["a-tag"]}))


def test_metadata_is_merged() -> None:
    """Test metadata and tags defined in with_config and at are merged/concatend."""

    foo = RunnableLambda(lambda x: x).with_config({"metadata": {"my_key": "my_value"}})
    expected_metadata = {
        "my_key": "my_value",
        "my_other_key": "my_other_value",
    }
    with collect_runs() as cb:
        foo.invoke("hi", {"metadata": {"my_other_key": "my_other_value"}})
        run = cb.traced_runs[0]
    assert run.extra is not None
    assert run.extra["metadata"] == expected_metadata


def test_tags_are_appended() -> None:
    """Test tags from with_config are concatenated with those in invocation."""

    foo = RunnableLambda(lambda x: x).with_config({"tags": ["my_key"]})
    with collect_runs() as cb:
        foo.invoke("hi", {"tags": ["invoked_key"]})
        run = cb.traced_runs[0]
    assert isinstance(run.tags, list)
    assert sorted(run.tags) == sorted(["my_key", "invoked_key"])


def test_bind_bind() -> None:
    llm = FakeListLLM(responses=["i'm a textbot"])

    assert dumpd(
        llm.bind(stop=["Thought:"], one="two").bind(
            stop=["Observation:"], hello="world"
        )
    ) == dumpd(llm.bind(stop=["Observation:"], one="two", hello="world"))


def test_bind_with_lambda() -> None:
    def my_function(*args: Any, **kwargs: Any) -> int:
        return 3 + kwargs.get("n", 0)

    runnable = RunnableLambda(my_function).bind(n=1)
    assert 4 == runnable.invoke({})
    chunks = list(runnable.stream({}))
    assert [4] == chunks


async def test_bind_with_lambda_async() -> None:
    def my_function(*args: Any, **kwargs: Any) -> int:
        return 3 + kwargs.get("n", 0)

    runnable = RunnableLambda(my_function).bind(n=1)
    assert 4 == await runnable.ainvoke({})
    chunks = [item async for item in runnable.astream({})]
    assert [4] == chunks


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


def test_deep_stream_assign() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain: Runnable = prompt | llm | {"str": StrOutputParser()}

    stream = chain.stream({"question": "What up"})

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert add(chunks) == {"str": "foo-lish"}

    chain_with_assign = chain.assign(hello=itemgetter("str") | llm)

    assert chain_with_assign.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"question": {"title": "Question", "type": "string"}},
    }
    assert chain_with_assign.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "object",
        "properties": {
            "str": {"title": "Str", "type": "string"},
            "hello": {"title": "Hello", "type": "string"},
        },
    }

    chunks = []
    for chunk in chain_with_assign.stream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish") * 2
    assert chunks == [
        # first stream passthrough input chunks
        {"str": "f"},
        {"str": "o"},
        {"str": "o"},
        {"str": "-"},
        {"str": "l"},
        {"str": "i"},
        {"str": "s"},
        {"str": "h"},
        # then stream assign output chunks
        {"hello": "f"},
        {"hello": "o"},
        {"hello": "o"},
        {"hello": "-"},
        {"hello": "l"},
        {"hello": "i"},
        {"hello": "s"},
        {"hello": "h"},
    ]
    assert add(chunks) == {"str": "foo-lish", "hello": "foo-lish"}
    assert chain_with_assign.invoke({"question": "What up"}) == {
        "str": "foo-lish",
        "hello": "foo-lish",
    }

    chain_with_assign_shadow = chain.assign(
        str=lambda _: "shadow",
        hello=itemgetter("str") | llm,
    )

    assert chain_with_assign_shadow.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"question": {"title": "Question", "type": "string"}},
    }
    assert chain_with_assign_shadow.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "object",
        "properties": {
            "str": {"title": "Str"},
            "hello": {"title": "Hello", "type": "string"},
        },
    }

    chunks = []
    for chunk in chain_with_assign_shadow.stream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish") + 1
    assert add(chunks) == {"str": "shadow", "hello": "foo-lish"}
    assert chain_with_assign_shadow.invoke({"question": "What up"}) == {
        "str": "shadow",
        "hello": "foo-lish",
    }


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


async def test_deep_astream_assign() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain: Runnable = prompt | llm | {"str": StrOutputParser()}

    stream = chain.astream({"question": "What up"})

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert add(chunks) == {"str": "foo-lish"}

    chain_with_assign = chain.assign(
        hello=itemgetter("str") | llm,
    )

    assert chain_with_assign.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"question": {"title": "Question", "type": "string"}},
    }
    assert chain_with_assign.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "object",
        "properties": {
            "str": {"title": "Str", "type": "string"},
            "hello": {"title": "Hello", "type": "string"},
        },
    }

    chunks = []
    async for chunk in chain_with_assign.astream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish") * 2
    assert chunks == [
        # first stream passthrough input chunks
        {"str": "f"},
        {"str": "o"},
        {"str": "o"},
        {"str": "-"},
        {"str": "l"},
        {"str": "i"},
        {"str": "s"},
        {"str": "h"},
        # then stream assign output chunks
        {"hello": "f"},
        {"hello": "o"},
        {"hello": "o"},
        {"hello": "-"},
        {"hello": "l"},
        {"hello": "i"},
        {"hello": "s"},
        {"hello": "h"},
    ]
    assert add(chunks) == {"str": "foo-lish", "hello": "foo-lish"}
    assert await chain_with_assign.ainvoke({"question": "What up"}) == {
        "str": "foo-lish",
        "hello": "foo-lish",
    }

    chain_with_assign_shadow = chain | RunnablePassthrough.assign(
        str=lambda _: "shadow",
        hello=itemgetter("str") | llm,
    )

    assert chain_with_assign_shadow.input_schema.schema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"question": {"title": "Question", "type": "string"}},
    }
    assert chain_with_assign_shadow.output_schema.schema() == {
        "title": "RunnableSequenceOutput",
        "type": "object",
        "properties": {
            "str": {"title": "Str"},
            "hello": {"title": "Hello", "type": "string"},
        },
    }

    chunks = []
    async for chunk in chain_with_assign_shadow.astream({"question": "What up"}):
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish") + 1
    assert add(chunks) == {"str": "shadow", "hello": "foo-lish"}
    assert await chain_with_assign_shadow.ainvoke({"question": "What up"}) == {
        "str": "shadow",
        "hello": "foo-lish",
    }


def test_runnable_sequence_transform() -> None:
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain: Runnable = llm | StrOutputParser()

    stream = chain.transform(llm.stream("Hi there!"))

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"


async def test_runnable_sequence_atransform() -> None:
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain: Runnable = llm | StrOutputParser()

    stream = chain.atransform(llm.astream("Hi there!"))

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == len("foo-lish")
    assert "".join(chunks) == "foo-lish"


class FakeSplitIntoListParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return True

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz`"
        )

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


def test_each_simple() -> None:
    """Test that each() works with a simple runnable."""
    parser = FakeSplitIntoListParser()
    assert parser.invoke("first item, second item") == ["first item", "second item"]
    assert parser.map().invoke(["a, b", "c"]) == [["a", "b"], ["c"]]
    assert parser.map().map().invoke([["a, b", "c"], ["c, e"]]) == [
        [["a", "b"], ["c"]],
        [["c", "e"]],
    ]


def test_each(snapshot: SnapshotAssertion) -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    first_llm = FakeStreamingListLLM(responses=["first item, second item, third item"])
    parser = FakeSplitIntoListParser()
    second_llm = FakeStreamingListLLM(responses=["this", "is", "a", "test"])

    chain = prompt | first_llm | parser | second_llm.map()

    assert dumps(chain, pretty=True) == snapshot
    output = chain.invoke({"question": "What up"})
    assert output == ["this", "is", "a"]

    assert (parser | second_llm.map()).invoke("first item, second item") == [
        "test",
        "this",
    ]


def test_recursive_lambda() -> None:
    def _simple_recursion(x: int) -> Union[int, Runnable]:
        if x < 10:
            return RunnableLambda(lambda *args: _simple_recursion(x + 1))
        else:
            return x

    runnable = RunnableLambda(_simple_recursion)
    assert runnable.invoke(5) == 10

    with pytest.raises(RecursionError):
        runnable.invoke(0, {"recursion_limit": 9})


def test_retrying(mocker: MockerFixture) -> None:
    def _lambda(x: int) -> Union[int, Runnable]:
        if x == 1:
            raise ValueError("x is 1")
        elif x == 2:
            raise RuntimeError("x is 2")
        else:
            return x

    _lambda_mock = mocker.Mock(side_effect=_lambda)
    runnable = RunnableLambda(_lambda_mock)

    with pytest.raises(ValueError):
        runnable.invoke(1)

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        runnable.with_retry(
            stop_after_attempt=2,
            retry_if_exception_type=(ValueError,),
        ).invoke(1)

    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()

    with pytest.raises(RuntimeError):
        runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).invoke(2)

    assert _lambda_mock.call_count == 1  # did not retry
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).batch([1, 2, 0])

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    _lambda_mock.reset_mock()

    output = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    ).batch([1, 2, 0], return_exceptions=True)

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    assert len(output) == 3
    assert isinstance(output[0], ValueError)
    assert isinstance(output[1], RuntimeError)
    assert output[2] == 0
    _lambda_mock.reset_mock()


async def test_async_retrying(mocker: MockerFixture) -> None:
    def _lambda(x: int) -> Union[int, Runnable]:
        if x == 1:
            raise ValueError("x is 1")
        elif x == 2:
            raise RuntimeError("x is 2")
        else:
            return x

    _lambda_mock = mocker.Mock(side_effect=_lambda)
    runnable = RunnableLambda(_lambda_mock)

    with pytest.raises(ValueError):
        await runnable.ainvoke(1)

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError, KeyError),
        ).ainvoke(1)

    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()

    with pytest.raises(RuntimeError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).ainvoke(2)

    assert _lambda_mock.call_count == 1  # did not retry
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).abatch([1, 2, 0])

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    _lambda_mock.reset_mock()

    output = await runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    ).abatch([1, 2, 0], return_exceptions=True)

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    assert len(output) == 3
    assert isinstance(output[0], ValueError)
    assert isinstance(output[1], RuntimeError)
    assert output[2] == 0
    _lambda_mock.reset_mock()


def test_runnable_lambda_stream() -> None:
    """Test that stream works for both normal functions & those returning Runnable."""
    # Normal output should work
    output: List[Any] = [chunk for chunk in RunnableLambda(range).stream(5)]
    assert output == [range(5)]

    # Runnable output should also work
    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    output = list(RunnableLambda(lambda x: llm).stream(""))
    assert output == list(llm_res)


def test_runnable_lambda_stream_with_callbacks() -> None:
    """Test that stream works for RunnableLambda when using callbacks."""
    tracer = FakeTracer()

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)
    config: RunnableConfig = {"callbacks": [tracer]}

    assert list(RunnableLambda(lambda x: llm).stream("", config=config)) == list(
        llm_res
    )

    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": llm_res}

    def raise_value_error(x: int) -> int:
        """Raise a value error."""
        raise ValueError("x is too large")

    # Check that the chain on error is invoked
    with pytest.raises(ValueError):
        for _ in RunnableLambda(raise_value_error).stream(1000, config=config):
            pass

    assert len(tracer.runs) == 2
    assert "ValueError('x is too large')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None


async def test_runnable_lambda_astream() -> None:
    """Test that astream works for both normal functions & those returning Runnable."""

    # Wrapper to make a normal function async
    def awrapper(func: Callable) -> Callable[..., Awaitable[Any]]:
        async def afunc(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return afunc

    # Normal output should work
    output: List[Any] = [
        chunk
        async for chunk in RunnableLambda(
            func=id,
            afunc=awrapper(range),  # id func is just dummy
        ).astream(5)
    ]
    assert output == [range(5)]

    # Normal output using func should also work
    output = [_ async for _ in RunnableLambda(range).astream(5)]
    assert output == [range(5)]

    # Runnable output should also work
    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    output = [
        _
        async for _ in RunnableLambda(
            func=id,
            afunc=awrapper(lambda x: llm),
        ).astream("")
    ]
    assert output == list(llm_res)

    output = [
        chunk
        async for chunk in cast(
            AsyncIterator[str], RunnableLambda(lambda x: llm).astream("")
        )
    ]
    assert output == list(llm_res)


async def test_runnable_lambda_astream_with_callbacks() -> None:
    """Test that astream works for RunnableLambda when using callbacks."""
    tracer = FakeTracer()

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)
    config: RunnableConfig = {"callbacks": [tracer]}

    assert [
        _ async for _ in RunnableLambda(lambda x: llm).astream("", config=config)
    ] == list(llm_res)

    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": llm_res}

    def raise_value_error(x: int) -> int:
        """Raise a value error."""
        raise ValueError("x is too large")

    # Check that the chain on error is invoked
    with pytest.raises(ValueError):
        async for _ in RunnableLambda(raise_value_error).astream(1000, config=config):
            pass

    assert len(tracer.runs) == 2
    assert "ValueError('x is too large')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None


@freeze_time("2023-01-01")
def test_seq_batch_return_exceptions(mocker: MockerFixture) -> None:
    class ControlledExceptionRunnable(Runnable[str, str]):
        def __init__(self, fail_starts_with: str) -> None:
            self.fail_starts_with = fail_starts_with

        def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
            raise NotImplementedError()

        def _batch(
            self,
            inputs: List[str],
        ) -> List:
            outputs: List[Any] = []
            for input in inputs:
                if input.startswith(self.fail_starts_with):
                    outputs.append(ValueError())
                else:
                    outputs.append(input + "a")
            return outputs

        def batch(
            self,
            inputs: List[str],
            config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
            *,
            return_exceptions: bool = False,
            **kwargs: Any,
        ) -> List[str]:
            return self._batch_with_config(
                self._batch,
                inputs,
                config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    chain = (
        ControlledExceptionRunnable("bux")
        | ControlledExceptionRunnable("bar")
        | ControlledExceptionRunnable("baz")
        | ControlledExceptionRunnable("foo")
    )

    assert isinstance(chain, RunnableSequence)

    # Test batch
    with pytest.raises(ValueError):
        chain.batch(["foo", "bar", "baz", "qux"])

    spy = mocker.spy(ControlledExceptionRunnable, "batch")
    tracer = FakeTracer()
    inputs = ["foo", "bar", "baz", "qux"]
    outputs = chain.batch(inputs, dict(callbacks=[tracer]), return_exceptions=True)
    assert len(outputs) == 4
    assert isinstance(outputs[0], ValueError)
    assert isinstance(outputs[1], ValueError)
    assert isinstance(outputs[2], ValueError)
    assert outputs[3] == "quxaaaa"
    assert spy.call_count == 4
    inputs_to_batch = [c[0][1] for c in spy.call_args_list]
    assert inputs_to_batch == [
        # inputs to sequence step 0
        # same as inputs to sequence.batch()
        ["foo", "bar", "baz", "qux"],
        # inputs to sequence step 1
        # == outputs of sequence step 0 as no exceptions were raised
        ["fooa", "bara", "baza", "quxa"],
        # inputs to sequence step 2
        # 'bar' was dropped as it raised an exception in step 1
        ["fooaa", "bazaa", "quxaa"],
        # inputs to sequence step 3
        # 'baz' was dropped as it raised an exception in step 2
        ["fooaaa", "quxaaa"],
    ]
    parent_runs = sorted(
        (r for r in tracer.runs if r.parent_run_id is None),
        key=lambda run: inputs.index(run.inputs["input"]),
    )
    assert len(parent_runs) == 4

    parent_run_foo = parent_runs[0]
    assert parent_run_foo.inputs["input"] == "foo"
    assert repr(ValueError()) in str(parent_run_foo.error)
    assert len(parent_run_foo.child_runs) == 4
    assert [r.error for r in parent_run_foo.child_runs[:-1]] == [
        None,
        None,
        None,
    ]
    assert repr(ValueError()) in str(parent_run_foo.child_runs[-1].error)

    parent_run_bar = parent_runs[1]
    assert parent_run_bar.inputs["input"] == "bar"
    assert repr(ValueError()) in str(parent_run_bar.error)
    assert len(parent_run_bar.child_runs) == 2
    assert parent_run_bar.child_runs[0].error is None
    assert repr(ValueError()) in str(parent_run_bar.child_runs[1].error)

    parent_run_baz = parent_runs[2]
    assert parent_run_baz.inputs["input"] == "baz"
    assert repr(ValueError()) in str(parent_run_baz.error)
    assert len(parent_run_baz.child_runs) == 3

    assert [r.error for r in parent_run_baz.child_runs[:-1]] == [
        None,
        None,
    ]
    assert repr(ValueError()) in str(parent_run_baz.child_runs[-1].error)

    parent_run_qux = parent_runs[3]
    assert parent_run_qux.inputs["input"] == "qux"
    assert parent_run_qux.error is None
    assert parent_run_qux.outputs is not None
    assert parent_run_qux.outputs["output"] == "quxaaaa"
    assert len(parent_run_qux.child_runs) == 4
    assert [r.error for r in parent_run_qux.child_runs] == [None, None, None, None]


@freeze_time("2023-01-01")
async def test_seq_abatch_return_exceptions(mocker: MockerFixture) -> None:
    class ControlledExceptionRunnable(Runnable[str, str]):
        def __init__(self, fail_starts_with: str) -> None:
            self.fail_starts_with = fail_starts_with

        def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
            raise NotImplementedError()

        async def _abatch(
            self,
            inputs: List[str],
        ) -> List:
            outputs: List[Any] = []
            for input in inputs:
                if input.startswith(self.fail_starts_with):
                    outputs.append(ValueError())
                else:
                    outputs.append(input + "a")
            return outputs

        async def abatch(
            self,
            inputs: List[str],
            config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
            *,
            return_exceptions: bool = False,
            **kwargs: Any,
        ) -> List[str]:
            return await self._abatch_with_config(
                self._abatch,
                inputs,
                config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    chain = (
        ControlledExceptionRunnable("bux")
        | ControlledExceptionRunnable("bar")
        | ControlledExceptionRunnable("baz")
        | ControlledExceptionRunnable("foo")
    )

    assert isinstance(chain, RunnableSequence)

    # Test abatch
    with pytest.raises(ValueError):
        await chain.abatch(["foo", "bar", "baz", "qux"])

    spy = mocker.spy(ControlledExceptionRunnable, "abatch")
    tracer = FakeTracer()
    inputs = ["foo", "bar", "baz", "qux"]
    outputs = await chain.abatch(
        inputs, dict(callbacks=[tracer]), return_exceptions=True
    )
    assert len(outputs) == 4
    assert isinstance(outputs[0], ValueError)
    assert isinstance(outputs[1], ValueError)
    assert isinstance(outputs[2], ValueError)
    assert outputs[3] == "quxaaaa"
    assert spy.call_count == 4
    inputs_to_batch = [c[0][1] for c in spy.call_args_list]
    assert inputs_to_batch == [
        # inputs to sequence step 0
        # same as inputs to sequence.batch()
        ["foo", "bar", "baz", "qux"],
        # inputs to sequence step 1
        # == outputs of sequence step 0 as no exceptions were raised
        ["fooa", "bara", "baza", "quxa"],
        # inputs to sequence step 2
        # 'bar' was dropped as it raised an exception in step 1
        ["fooaa", "bazaa", "quxaa"],
        # inputs to sequence step 3
        # 'baz' was dropped as it raised an exception in step 2
        ["fooaaa", "quxaaa"],
    ]
    parent_runs = sorted(
        (r for r in tracer.runs if r.parent_run_id is None),
        key=lambda run: inputs.index(run.inputs["input"]),
    )
    assert len(parent_runs) == 4

    parent_run_foo = parent_runs[0]
    assert parent_run_foo.inputs["input"] == "foo"
    assert repr(ValueError()) in str(parent_run_foo.error)
    assert len(parent_run_foo.child_runs) == 4
    assert [r.error for r in parent_run_foo.child_runs[:-1]] == [
        None,
        None,
        None,
    ]
    assert repr(ValueError()) in str(parent_run_foo.child_runs[-1].error)

    parent_run_bar = parent_runs[1]
    assert parent_run_bar.inputs["input"] == "bar"
    assert repr(ValueError()) in str(parent_run_bar.error)
    assert len(parent_run_bar.child_runs) == 2
    assert parent_run_bar.child_runs[0].error is None
    assert repr(ValueError()) in str(parent_run_bar.child_runs[1].error)

    parent_run_baz = parent_runs[2]
    assert parent_run_baz.inputs["input"] == "baz"
    assert repr(ValueError()) in str(parent_run_baz.error)
    assert len(parent_run_baz.child_runs) == 3
    assert [r.error for r in parent_run_baz.child_runs[:-1]] == [
        None,
        None,
    ]
    assert repr(ValueError()) in str(parent_run_baz.child_runs[-1].error)

    parent_run_qux = parent_runs[3]
    assert parent_run_qux.inputs["input"] == "qux"
    assert parent_run_qux.error is None
    assert parent_run_qux.outputs is not None
    assert parent_run_qux.outputs["output"] == "quxaaaa"
    assert len(parent_run_qux.child_runs) == 4
    assert [r.error for r in parent_run_qux.child_runs] == [None, None, None, None]


def test_runnable_branch_init() -> None:
    """Verify that runnable branch gets initialized properly."""
    add = RunnableLambda(lambda x: x + 1)
    condition = RunnableLambda(lambda x: x > 0)

    # Test failure with less than 2 branches
    with pytest.raises(ValueError):
        RunnableBranch((condition, add))

    # Test failure with less than 2 branches
    with pytest.raises(ValueError):
        RunnableBranch(condition)


@pytest.mark.parametrize(
    "branches",
    [
        [
            (RunnableLambda(lambda x: x > 0), RunnableLambda(lambda x: x + 1)),
            RunnableLambda(lambda x: x - 1),
        ],
        [
            (RunnableLambda(lambda x: x > 0), RunnableLambda(lambda x: x + 1)),
            (RunnableLambda(lambda x: x > 5), RunnableLambda(lambda x: x + 1)),
            RunnableLambda(lambda x: x - 1),
        ],
        [
            (lambda x: x > 0, lambda x: x + 1),
            (lambda x: x > 5, lambda x: x + 1),
            lambda x: x - 1,
        ],
    ],
)
def test_runnable_branch_init_coercion(branches: Sequence[Any]) -> None:
    """Verify that runnable branch gets initialized properly."""
    runnable = RunnableBranch[int, int](*branches)
    for branch in runnable.branches:
        condition, body = branch
        assert isinstance(condition, Runnable)
        assert isinstance(body, Runnable)

    assert isinstance(runnable.default, Runnable)
    assert runnable.input_schema.schema() == {"title": "RunnableBranchInput"}


def test_runnable_branch_invoke_call_counts(mocker: MockerFixture) -> None:
    """Verify that runnables are invoked only when necessary."""
    # Test with single branch
    add = RunnableLambda(lambda x: x + 1)
    sub = RunnableLambda(lambda x: x - 1)
    condition = RunnableLambda(lambda x: x > 0)
    spy = mocker.spy(condition, "invoke")
    add_spy = mocker.spy(add, "invoke")

    branch = RunnableBranch[int, int]((condition, add), (condition, add), sub)
    assert spy.call_count == 0
    assert add_spy.call_count == 0

    assert branch.invoke(1) == 2
    assert add_spy.call_count == 1
    assert spy.call_count == 1

    assert branch.invoke(2) == 3
    assert spy.call_count == 2
    assert add_spy.call_count == 2

    assert branch.invoke(-3) == -4
    # Should fall through to default branch with condition being evaluated twice!
    assert spy.call_count == 4
    # Add should not be invoked
    assert add_spy.call_count == 2


def test_runnable_branch_invoke() -> None:
    # Test with single branch
    def raise_value_error(x: int) -> int:
        """Raise a value error."""
        raise ValueError("x is too large")

    branch = RunnableBranch[int, int](
        (lambda x: x > 100, raise_value_error),
        # mypy cannot infer types from the lambda
        (lambda x: x > 0 and x < 5, lambda x: x + 1),  # type: ignore[misc]
        (lambda x: x > 5, lambda x: x * 10),
        lambda x: x - 1,
    )

    assert branch.invoke(1) == 2
    assert branch.invoke(10) == 100
    assert branch.invoke(0) == -1
    # Should raise an exception
    with pytest.raises(ValueError):
        branch.invoke(1000)


def test_runnable_branch_batch() -> None:
    """Test batch variant."""
    # Test with single branch
    branch = RunnableBranch[int, int](
        (lambda x: x > 0 and x < 5, lambda x: x + 1),
        (lambda x: x > 5, lambda x: x * 10),
        lambda x: x - 1,
    )

    assert branch.batch([1, 10, 0]) == [2, 100, -1]


async def test_runnable_branch_ainvoke() -> None:
    """Test async variant of invoke."""
    branch = RunnableBranch[int, int](
        (lambda x: x > 0 and x < 5, lambda x: x + 1),
        (lambda x: x > 5, lambda x: x * 10),
        lambda x: x - 1,
    )

    assert await branch.ainvoke(1) == 2
    assert await branch.ainvoke(10) == 100
    assert await branch.ainvoke(0) == -1

    # Verify that the async variant is used if available
    async def condition(x: int) -> bool:
        return x > 0

    async def add(x: int) -> int:
        return x + 1

    async def sub(x: int) -> int:
        return x - 1

    branch = RunnableBranch[int, int]((condition, add), sub)

    assert await branch.ainvoke(1) == 2
    assert await branch.ainvoke(-10) == -11


def test_runnable_branch_invoke_callbacks() -> None:
    """Verify that callbacks are correctly used in invoke."""
    tracer = FakeTracer()

    def raise_value_error(x: int) -> int:
        """Raise a value error."""
        raise ValueError("x is too large")

    branch = RunnableBranch[int, int](
        (lambda x: x > 100, raise_value_error),
        lambda x: x - 1,
    )

    assert branch.invoke(1, config={"callbacks": [tracer]}) == 0
    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": 0}

    # Check that the chain on end is invoked
    with pytest.raises(ValueError):
        branch.invoke(1000, config={"callbacks": [tracer]})

    assert len(tracer.runs) == 2
    assert "ValueError('x is too large')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None


async def test_runnable_branch_ainvoke_callbacks() -> None:
    """Verify that callbacks are invoked correctly in ainvoke."""
    tracer = FakeTracer()

    async def raise_value_error(x: int) -> int:
        """Raise a value error."""
        raise ValueError("x is too large")

    branch = RunnableBranch[int, int](
        (lambda x: x > 100, raise_value_error),
        lambda x: x - 1,
    )

    assert await branch.ainvoke(1, config={"callbacks": [tracer]}) == 0
    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": 0}

    # Check that the chain on end is invoked
    with pytest.raises(ValueError):
        await branch.ainvoke(1000, config={"callbacks": [tracer]})

    assert len(tracer.runs) == 2
    assert "ValueError('x is too large')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None


async def test_runnable_branch_abatch() -> None:
    """Test async variant of invoke."""
    branch = RunnableBranch[int, int](
        (lambda x: x > 0 and x < 5, lambda x: x + 1),
        (lambda x: x > 5, lambda x: x * 10),
        lambda x: x - 1,
    )

    assert await branch.abatch([1, 10, 0]) == [2, 100, -1]


def test_runnable_branch_stream() -> None:
    """Verify that stream works for RunnableBranch."""

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    branch = RunnableBranch[str, Any](
        (lambda x: x == "hello", llm),
        lambda x: x,
    )

    assert list(branch.stream("hello")) == list(llm_res)
    assert list(branch.stream("bye")) == ["bye"]


def test_runnable_branch_stream_with_callbacks() -> None:
    """Verify that stream works for RunnableBranch when using callbacks."""
    tracer = FakeTracer()

    def raise_value_error(x: str) -> Any:
        """Raise a value error."""
        raise ValueError(f"x is {x}")

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    branch = RunnableBranch[str, Any](
        (lambda x: x == "error", raise_value_error),
        (lambda x: x == "hello", llm),
        lambda x: x,
    )
    config: RunnableConfig = {"callbacks": [tracer]}

    assert list(branch.stream("hello", config=config)) == list(llm_res)

    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": llm_res}

    # Verify that the chain on error is invoked
    with pytest.raises(ValueError):
        for _ in branch.stream("error", config=config):
            pass

    assert len(tracer.runs) == 2
    assert "ValueError('x is error')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None

    assert list(branch.stream("bye", config=config)) == ["bye"]

    assert len(tracer.runs) == 3
    assert tracer.runs[2].error is None
    assert tracer.runs[2].outputs == {"output": "bye"}


async def test_runnable_branch_astream() -> None:
    """Verify that astream works for RunnableBranch."""

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    branch = RunnableBranch[str, Any](
        (lambda x: x == "hello", llm),
        lambda x: x,
    )

    assert [_ async for _ in branch.astream("hello")] == list(llm_res)
    assert [_ async for _ in branch.astream("bye")] == ["bye"]

    # Verify that the async variant is used if available
    async def condition(x: str) -> bool:
        return x == "hello"

    async def repeat(x: str) -> str:
        return x + x

    async def reverse(x: str) -> str:
        return x[::-1]

    branch = RunnableBranch[str, Any]((condition, repeat), llm)

    assert [_ async for _ in branch.astream("hello")] == ["hello" * 2]
    assert [_ async for _ in branch.astream("bye")] == list(llm_res)

    branch = RunnableBranch[str, Any]((condition, llm), reverse)

    assert [_ async for _ in branch.astream("hello")] == list(llm_res)
    assert [_ async for _ in branch.astream("bye")] == ["eyb"]


async def test_runnable_branch_astream_with_callbacks() -> None:
    """Verify that astream works for RunnableBranch when using callbacks."""
    tracer = FakeTracer()

    def raise_value_error(x: str) -> Any:
        """Raise a value error."""
        raise ValueError(f"x is {x}")

    llm_res = "i'm a textbot"
    # sleep to better simulate a real stream
    llm = FakeStreamingListLLM(responses=[llm_res], sleep=0.01)

    branch = RunnableBranch[str, Any](
        (lambda x: x == "error", raise_value_error),
        (lambda x: x == "hello", llm),
        lambda x: x,
    )
    config: RunnableConfig = {"callbacks": [tracer]}

    assert [_ async for _ in branch.astream("hello", config=config)] == list(llm_res)

    assert len(tracer.runs) == 1
    assert tracer.runs[0].error is None
    assert tracer.runs[0].outputs == {"output": llm_res}

    # Verify that the chain on error is invoked
    with pytest.raises(ValueError):
        async for _ in branch.astream("error", config=config):
            pass

    assert len(tracer.runs) == 2
    assert "ValueError('x is error')" in str(tracer.runs[1].error)
    assert tracer.runs[1].outputs is None

    assert [_ async for _ in branch.astream("bye", config=config)] == ["bye"]

    assert len(tracer.runs) == 3
    assert tracer.runs[2].error is None
    assert tracer.runs[2].outputs == {"output": "bye"}


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
def test_representation_of_runnables() -> None:
    """Test representation of runnables."""
    runnable = RunnableLambda(lambda x: x * 2)
    assert repr(runnable) == "RunnableLambda(lambda x: x * 2)"

    def f(x: int) -> int:
        """Return 2."""
        return 2

    assert repr(RunnableLambda(func=f)) == "RunnableLambda(f)"

    async def af(x: int) -> int:
        """Return 2."""
        return 2

    assert repr(RunnableLambda(func=f, afunc=af)) == "RunnableLambda(f)"

    assert repr(
        RunnableLambda(lambda x: x + 2)
        | {
            "a": RunnableLambda(lambda x: x * 2),
            "b": RunnableLambda(lambda x: x * 3),
        }
    ) == (
        "RunnableLambda(...)\n"
        "| {\n"
        "    a: RunnableLambda(...),\n"
        "    b: RunnableLambda(...)\n"
        "  }"
    ), "repr where code string contains multiple lambdas gives up"


async def test_tool_from_runnable() -> None:
    prompt = (
        SystemMessagePromptTemplate.from_template("You are a nice assistant.")
        + "{question}"
    )
    llm = FakeStreamingListLLM(responses=["foo-lish"])

    chain = prompt | llm | StrOutputParser()

    chain_tool = tool("chain_tool", chain)

    assert isinstance(chain_tool, BaseTool)
    assert chain_tool.name == "chain_tool"
    assert chain_tool.run({"question": "What up"}) == chain.invoke(
        {"question": "What up"}
    )
    assert await chain_tool.arun({"question": "What up"}) == await chain.ainvoke(
        {"question": "What up"}
    )
    assert chain_tool.description.endswith(repr(chain))
    assert chain_tool.args_schema.schema() == chain.input_schema.schema()
    assert chain_tool.args_schema.schema() == {
        "properties": {"question": {"title": "Question", "type": "string"}},
        "title": "PromptInput",
        "type": "object",
    }


async def test_runnable_gen() -> None:
    """Test that a generator can be used as a runnable."""

    def gen(input: Iterator[Any]) -> Iterator[int]:
        yield 1
        yield 2
        yield 3

    runnable = RunnableGenerator(gen)

    assert runnable.input_schema.schema() == {"title": "gen_input"}
    assert runnable.output_schema.schema() == {
        "title": "gen_output",
        "type": "integer",
    }

    assert runnable.invoke(None) == 6
    assert list(runnable.stream(None)) == [1, 2, 3]
    assert runnable.batch([None, None]) == [6, 6]

    async def agen(input: AsyncIterator[Any]) -> AsyncIterator[int]:
        yield 1
        yield 2
        yield 3

    arunnable = RunnableGenerator(agen)

    assert await arunnable.ainvoke(None) == 6
    assert [p async for p in arunnable.astream(None)] == [1, 2, 3]
    assert await arunnable.abatch([None, None]) == [6, 6]


async def test_runnable_gen_context_config() -> None:
    """Test that a generator can call other runnables with config
    propagated from the context."""

    fake = RunnableLambda(len)

    def gen(input: Iterator[Any]) -> Iterator[int]:
        yield fake.invoke("a")
        yield fake.invoke("aa")
        yield fake.invoke("aaa")

    runnable = RunnableGenerator(gen)

    assert runnable.input_schema.schema() == {"title": "gen_input"}
    assert runnable.output_schema.schema() == {
        "title": "gen_output",
        "type": "integer",
    }

    tracer = FakeTracer()
    assert runnable.invoke(None, {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert list(runnable.stream(None)) == [1, 2, 3]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert list(runnable.stream(None, {"callbacks": [tracer]})) == [1, 2, 3]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert runnable.batch([None, None], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]

    if sys.version_info < (3, 11):
        # Python 3.10 and below don't support running async tasks in a specific context
        return

    async def agen(input: AsyncIterator[Any]) -> AsyncIterator[int]:
        yield await fake.ainvoke("a")
        yield await fake.ainvoke("aa")
        yield await fake.ainvoke("aaa")

    arunnable = RunnableGenerator(agen)

    tracer = FakeTracer()
    assert await arunnable.ainvoke(None, {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert [p async for p in arunnable.astream(None)] == [1, 2, 3]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert [p async for p in arunnable.astream(None, {"callbacks": [tracer]})] == [
        1,
        2,
        3,
    ]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert await arunnable.abatch([None, None], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]


async def test_runnable_iter_context_config() -> None:
    """Test that a generator can call other runnables with config
    propagated from the context."""

    fake = RunnableLambda(len)

    @chain
    def gen(input: str) -> Iterator[int]:
        yield fake.invoke(input)
        yield fake.invoke(input * 2)
        yield fake.invoke(input * 3)

    assert gen.input_schema.schema() == {
        "title": "gen_input",
        "type": "string",
    }
    assert gen.output_schema.schema() == {
        "title": "gen_output",
        "type": "integer",
    }

    tracer = FakeTracer()
    assert gen.invoke("a", {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert list(gen.stream("a")) == [1, 2, 3]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert list(gen.stream("a", {"callbacks": [tracer]})) == [1, 2, 3]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert gen.batch(["a", "a"], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]

    if sys.version_info < (3, 11):
        # Python 3.10 and below don't support running async tasks in a specific context
        return

    @chain
    async def agen(input: str) -> AsyncIterator[int]:
        yield await fake.ainvoke(input)
        yield await fake.ainvoke(input * 2)
        yield await fake.ainvoke(input * 3)

    assert agen.input_schema.schema() == {
        "title": "agen_input",
        "type": "string",
    }
    assert agen.output_schema.schema() == {
        "title": "agen_output",
        "type": "integer",
    }

    tracer = FakeTracer()
    assert await agen.ainvoke("a", {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert [p async for p in agen.astream("a")] == [1, 2, 3]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert [p async for p in agen.astream("a", {"callbacks": [tracer]})] == [
        1,
        2,
        3,
    ]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert [p async for p in agen.astream_log("a", {"callbacks": [tracer]})]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert await agen.abatch(["a", "a"], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]


async def test_runnable_lambda_context_config() -> None:
    """Test that a function can call other runnables with config
    propagated from the context."""

    fake = RunnableLambda(len)

    @chain
    def fun(input: str) -> int:
        output = fake.invoke(input)
        output += fake.invoke(input * 2)
        output += fake.invoke(input * 3)
        return output

    assert fun.input_schema.schema() == {"title": "fun_input", "type": "string"}
    assert fun.output_schema.schema() == {
        "title": "fun_output",
        "type": "integer",
    }

    tracer = FakeTracer()
    assert fun.invoke("a", {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert list(fun.stream("a")) == [6]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert list(fun.stream("a", {"callbacks": [tracer]})) == [6]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert fun.batch(["a", "a"], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]

    if sys.version_info < (3, 11):
        # Python 3.10 and below don't support running async tasks in a specific context
        return

    @chain
    async def afun(input: str) -> int:
        output = await fake.ainvoke(input)
        output += await fake.ainvoke(input * 2)
        output += await fake.ainvoke(input * 3)
        return output

    assert afun.input_schema.schema() == {"title": "afun_input", "type": "string"}
    assert afun.output_schema.schema() == {
        "title": "afun_output",
        "type": "integer",
    }

    tracer = FakeTracer()
    assert await afun.ainvoke("a", {"callbacks": [tracer]}) == 6
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    tracer.runs.clear()

    assert [p async for p in afun.astream("a")] == [6]
    assert len(tracer.runs) == 0, "callbacks doesn't persist from previous call"

    tracer = FakeTracer()
    assert [p async for p in afun.astream("a", {"callbacks": [tracer]})] == [6]
    assert len(tracer.runs) == 1
    assert tracer.runs[0].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]

    tracer = FakeTracer()
    assert await afun.abatch(["a", "a"], {"callbacks": [tracer]}) == [6, 6]
    assert len(tracer.runs) == 2
    assert tracer.runs[0].outputs == {"output": 6}
    assert tracer.runs[1].outputs == {"output": 6}
    assert len(tracer.runs[0].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[0].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[0].child_runs] == [1, 2, 3]
    assert len(tracer.runs[1].child_runs) == 3
    assert [r.inputs["input"] for r in tracer.runs[1].child_runs] == ["a", "aa", "aaa"]
    assert [(r.outputs or {})["output"] for r in tracer.runs[1].child_runs] == [1, 2, 3]


async def test_runnable_gen_transform() -> None:
    """Test that a generator can be used as a runnable."""

    def gen_indexes(length_iter: Iterator[int]) -> Iterator[int]:
        for i in range(next(length_iter)):
            yield i

    async def agen_indexes(length_iter: AsyncIterator[int]) -> AsyncIterator[int]:
        async for length in length_iter:
            for i in range(length):
                yield i

    def plus_one(input: Iterator[int]) -> Iterator[int]:
        for i in input:
            yield i + 1

    async def aplus_one(input: AsyncIterator[int]) -> AsyncIterator[int]:
        async for i in input:
            yield i + 1

    chain: Runnable = RunnableGenerator(gen_indexes, agen_indexes) | plus_one
    achain = RunnableGenerator(gen_indexes, agen_indexes) | aplus_one

    assert chain.input_schema.schema() == {
        "title": "gen_indexes_input",
        "type": "integer",
    }
    assert chain.output_schema.schema() == {
        "title": "plus_one_output",
        "type": "integer",
    }
    assert achain.input_schema.schema() == {
        "title": "gen_indexes_input",
        "type": "integer",
    }
    assert achain.output_schema.schema() == {
        "title": "aplus_one_output",
        "type": "integer",
    }

    assert list(chain.stream(3)) == [1, 2, 3]
    assert [p async for p in achain.astream(4)] == [1, 2, 3, 4]


def test_with_config_callbacks() -> None:
    result = RunnableLambda(lambda x: x).with_config({"callbacks": []})
    # Bugfix from version 0.0.325
    # ConfigError: field "callbacks" not yet prepared so type is still a ForwardRef,
    # you might need to call RunnableConfig.update_forward_refs().
    assert isinstance(result, RunnableBinding)


async def test_ainvoke_on_returned_runnable() -> None:
    """Verify that a runnable returned by a sync runnable in the async path will
    be runthroughaasync path (issue #13407)"""

    def idchain_sync(__input: dict) -> bool:
        return False

    async def idchain_async(__input: dict) -> bool:
        return True

    idchain = RunnableLambda(func=idchain_sync, afunc=idchain_async)

    def func(__input: dict) -> Runnable:
        return idchain

    assert await RunnableLambda(func).ainvoke({})


def test_invoke_stream_passthrough_assign_trace() -> None:
    def idchain_sync(__input: dict) -> bool:
        return False

    chain = RunnablePassthrough.assign(urls=idchain_sync)

    tracer = FakeTracer()
    chain.invoke({"example": [1, 2, 3]}, dict(callbacks=[tracer]))

    assert tracer.runs[0].name == "RunnableAssign<urls>"
    assert tracer.runs[0].child_runs[0].name == "RunnableParallel<urls>"

    tracer = FakeTracer()
    for item in chain.stream({"example": [1, 2, 3]}, dict(callbacks=[tracer])):
        pass

    assert tracer.runs[0].name == "RunnableAssign<urls>"
    assert tracer.runs[0].child_runs[0].name == "RunnableParallel<urls>"


async def test_ainvoke_astream_passthrough_assign_trace() -> None:
    def idchain_sync(__input: dict) -> bool:
        return False

    chain = RunnablePassthrough.assign(urls=idchain_sync)

    tracer = FakeTracer()
    await chain.ainvoke({"example": [1, 2, 3]}, dict(callbacks=[tracer]))

    assert tracer.runs[0].name == "RunnableAssign<urls>"
    assert tracer.runs[0].child_runs[0].name == "RunnableParallel<urls>"

    tracer = FakeTracer()
    async for item in chain.astream({"example": [1, 2, 3]}, dict(callbacks=[tracer])):
        pass

    assert tracer.runs[0].name == "RunnableAssign<urls>"
    assert tracer.runs[0].child_runs[0].name == "RunnableParallel<urls>"


async def test_astream_log_deep_copies() -> None:
    """Verify that deep copies are used when using jsonpatch in astream log.

    jsonpatch re-uses objects in its API; e.g.,

    import jsonpatch
    obj1 = { "a": 1 }
    value = { "b": 2 }
    obj2 = { "a": 1, "value": value }

    ops = list(jsonpatch.JsonPatch.from_diff(obj1, obj2))
    assert id(ops[0]['value']) == id(value)

    This can create unexpected consequences for downstream code.
    """

    def _get_run_log(run_log_patches: Sequence[RunLogPatch]) -> RunLog:
        """Get run log"""
        run_log = RunLog(state=None)  # type: ignore
        for log_patch in run_log_patches:
            run_log = run_log + log_patch
        return run_log

    def add_one(x: int) -> int:
        """Add one."""
        return x + 1

    chain = RunnableLambda(add_one)
    chunks = []
    final_output = None
    async for chunk in chain.astream_log(1):
        chunks.append(chunk)
        if final_output is None:
            final_output = chunk
        else:
            final_output = final_output + chunk

    run_log = _get_run_log(chunks)
    state = run_log.state.copy()
    # Ignoring type here since we know that the state is a dict
    # so we can delete `id` for testing purposes
    state.pop("id")  # type: ignore
    assert state == {
        "final_output": 2,
        "logs": {},
        "streamed_output": [2],
        "name": "add_one",
        "type": "chain",
    }
