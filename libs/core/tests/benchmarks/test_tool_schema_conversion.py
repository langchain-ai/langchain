"""Benchmarks for tool-to-OpenAI schema conversion.

Agent loops convert every bound tool on every model call (token counting and
`bind_tools`), so conversion cost is paid per step. The warm benchmark measures
the steady state with the `tool_call_schema` memo populated; the cold benchmark
measures first-time conversion including subset-model creation.
"""

import pytest
from pydantic import BaseModel, Field, create_model
from pytest_benchmark.fixture import BenchmarkFixture

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

_NUM_TOOLS = 20
_NUM_FIELDS = 8


def _make_tools(num_tools: int) -> list[StructuredTool]:
    tools = []
    for i in range(num_tools):
        fields: dict = {
            f"param_{j}": (
                str | None,
                Field(default=None, description=f"Parameter {j} of action {i}."),
            )
            for j in range(_NUM_FIELDS)
        }
        fields["target"] = (str, Field(description="Primary target identifier."))
        schema: type[BaseModel] = create_model(f"BenchTool{i}Input", **fields)
        tools.append(
            StructuredTool.from_function(
                func=lambda **_kwargs: "ok",
                name=f"bench_tool_{i}",
                description=f"Benchmark tool {i} with several configurable options.",
                args_schema=schema,
            )
        )
    return tools


@pytest.mark.benchmark
def test_convert_tools_warm(benchmark: BenchmarkFixture) -> None:
    """Steady-state conversion of a reused toolset (memoized schema path)."""
    tools = _make_tools(_NUM_TOOLS)
    for tool in tools:
        convert_to_openai_tool(tool)  # populate the schema memo

    @benchmark  # type: ignore[untyped-decorator]
    def convert_warm() -> None:
        for tool in tools:
            convert_to_openai_tool(tool)


@pytest.mark.benchmark
def test_convert_tools_cold(benchmark: BenchmarkFixture) -> None:
    """First-time conversion, including subset-model creation per tool."""

    @benchmark  # type: ignore[untyped-decorator]
    def convert_cold() -> None:
        for tool in _make_tools(_NUM_TOOLS):
            convert_to_openai_tool(tool)
