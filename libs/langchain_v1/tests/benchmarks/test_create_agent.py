from __future__ import annotations

import tracemalloc
from itertools import cycle
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pytest_benchmark.fixture import BenchmarkFixture

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware import AgentMiddleware


# ---------------------------------------------------------------------------
# Tool fixtures — a realistic mix of simple, structured, and nested schemas
# ---------------------------------------------------------------------------


@tool
def simple_tool_1(x: int) -> str:
    """Add one to a number."""
    return str(x + 1)


@tool
def simple_tool_2(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@tool
def simple_tool_3(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def simple_tool_4(items: list[str]) -> int:
    """Count items in a list."""
    return len(items)


@tool
def simple_tool_5(flag: bool, value: str) -> str:
    """Return value if flag is true."""
    return value if flag else ""


class AddressSchema(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")


class PersonSchema(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    address: AddressSchema = Field(description="Home address")
    tags: list[str] = Field(default_factory=list, description="Tags")


@tool(args_schema=PersonSchema)
def structured_tool_1(name: str, age: int, address: AddressSchema, tags: list[str]) -> str:
    """Look up a person by their details."""
    return f"{name}, {age}"


class SearchSchema(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=10, description="Maximum results to return")
    filters: dict[str, str] = Field(default_factory=dict, description="Filter map")


@tool(args_schema=SearchSchema)
def structured_tool_2(query: str, max_results: int, filters: dict[str, str]) -> list[str]:
    """Search a database with filters."""
    return [query] * min(max_results, 3)


class FileSchema(BaseModel):
    path: str = Field(description="File path")
    encoding: str = Field(default="utf-8", description="File encoding")
    lines: list[int] = Field(default_factory=list, description="Line numbers to read")


@tool(args_schema=FileSchema)
def structured_tool_3(path: str, encoding: str, lines: list[int]) -> str:
    """Read a file at a given path."""
    return f"content of {path}"


class MatrixSchema(BaseModel):
    rows: int = Field(description="Number of rows")
    cols: int = Field(description="Number of columns")
    fill: float = Field(default=0.0, description="Fill value")


@tool(args_schema=MatrixSchema)
def structured_tool_4(rows: int, cols: int, fill: float) -> list[list[float]]:
    """Create a matrix filled with a value."""
    return [[fill] * cols for _ in range(rows)]


@tool
def complex_tool_1(
    name: str,
    metadata: dict[str, Any],
    tags: list[str],
    priority: int = 0,
) -> dict[str, Any]:
    """Create an item with metadata."""
    return {"name": name, "metadata": metadata, "tags": tags, "priority": priority}


@tool
def complex_tool_2(
    source: str,
    destination: str,
    options: dict[str, str] | None = None,
) -> bool:
    """Copy data from source to destination."""
    return True


@tool
def complex_tool_3(
    ids: list[int],
    batch_size: int = 10,
    retry: bool = False,
) -> dict[str, list[int]]:
    """Process a batch of IDs."""
    return {"processed": ids[:batch_size]}


@tool
def complex_tool_4(
    expression: str,
    variables: dict[str, float],
    precision: int = 6,
) -> float:
    """Evaluate a mathematical expression."""
    return 0.0


@tool
def complex_tool_5(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make an HTTP request."""
    return {"status": 200, "body": ""}


SMALL_TOOLS = [simple_tool_1, simple_tool_2, simple_tool_3]
MEDIUM_TOOLS = [
    simple_tool_1,
    simple_tool_2,
    simple_tool_3,
    simple_tool_4,
    simple_tool_5,
    structured_tool_1,
    structured_tool_2,
]
LARGE_TOOLS = [
    simple_tool_1,
    simple_tool_2,
    simple_tool_3,
    simple_tool_4,
    simple_tool_5,
    structured_tool_1,
    structured_tool_2,
    structured_tool_3,
    structured_tool_4,
    complex_tool_1,
    complex_tool_2,
    complex_tool_3,
    complex_tool_4,
    complex_tool_5,
]


def _make_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=cycle([AIMessage(content="ok")]))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_create_agent_instantiation(benchmark: BenchmarkFixture) -> None:
    """Baseline: no tools."""
    benchmark(lambda: create_agent(model=_make_model()))


@pytest.mark.benchmark
def test_create_agent_small_tools(benchmark: BenchmarkFixture) -> None:
    """3 simple tools."""
    benchmark(lambda: create_agent(model=_make_model(), tools=SMALL_TOOLS))


@pytest.mark.benchmark
def test_create_agent_medium_tools(benchmark: BenchmarkFixture) -> None:
    """7 mixed tools."""
    benchmark(lambda: create_agent(model=_make_model(), tools=MEDIUM_TOOLS))


@pytest.mark.benchmark
def test_create_agent_large_tools(benchmark: BenchmarkFixture) -> None:
    """14 tools including complex nested schemas."""
    benchmark(lambda: create_agent(model=_make_model(), tools=LARGE_TOOLS))


@pytest.mark.benchmark
def test_create_agent_large_tools_with_middleware(benchmark: BenchmarkFixture) -> None:
    """14 tools + full middleware stack."""
    middleware: Sequence[AgentMiddleware[Any, Any]] = (
        TodoListMiddleware(),
        ToolRetryMiddleware(),
        ModelRetryMiddleware(),
    )
    benchmark(
        lambda: create_agent(
            model=_make_model(),
            tools=LARGE_TOOLS,
            middleware=middleware,
        )
    )


@pytest.mark.benchmark
def test_tool_call_schema_repeated_access(benchmark: BenchmarkFixture) -> None:
    """Measure cost of repeated .tool_call_schema access on a complex tool."""
    t = structured_tool_1

    def access_schema_10x() -> None:
        for _ in range(10):
            _ = t.tool_call_schema

    benchmark(access_schema_10x)


@pytest.mark.benchmark
def test_tool_args_repeated_access(benchmark: BenchmarkFixture) -> None:
    """Measure cost of repeated .args access on a complex tool."""
    t = structured_tool_1

    def access_args_10x() -> None:
        for _ in range(10):
            _ = t.args

    benchmark(access_args_10x)


@pytest.mark.benchmark
def test_create_agent_instantiation_with_middleware(benchmark: BenchmarkFixture) -> None:
    """Baseline with middleware, no tools."""
    middleware: Sequence[AgentMiddleware[Any, Any]] = (
        TodoListMiddleware(),
        ToolRetryMiddleware(),
        ModelRetryMiddleware(),
    )
    benchmark(lambda: create_agent(model=_make_model(), middleware=middleware))


# ---------------------------------------------------------------------------
# Memory snapshot (not a codspeed benchmark — uses tracemalloc directly)
# ---------------------------------------------------------------------------


def test_create_agent_large_tools_memory() -> None:
    """Record peak memory for large-tools agent creation. Not a perf benchmark."""
    tracemalloc.start()
    create_agent(model=_make_model(), tools=LARGE_TOOLS)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Soft assertion: 50 MB is a generous ceiling for a single agent instantiation.
    assert peak < 50 * 1024 * 1024, f"Peak memory {peak / 1024 / 1024:.1f} MB exceeded 50 MB"
