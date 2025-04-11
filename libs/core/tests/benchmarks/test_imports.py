"""
Benchmarking import times for various langchain-core modules.

Benchmarking import times is a bit tricky - in order to create reproducible results
across runs, we need to avoid utilizing Python's import / modules cache.
Thus, we run the import in a subprocess.

At the moment, CodSpeed only supports [wall time](https://docs.codspeed.io/instruments/walltime/)
metrics for this benchmark (not the standard trace generation, etc).

Thus, we've temporarily marked this as a local only benchmark, though hopefully that
changes in the short term.
"""

import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]


@pytest.mark.local_benchmark
@pytest.mark.parametrize(
    "import_path",
    [
        pytest.param(
            "from langchain_core.messages import HumanMessage", id="HumanMessage"
        ),
        pytest.param("from langchain_core.tools import tool", id="tool"),
        pytest.param(
            "from langchain_core.callbacks import CallbackManager", id="CallbackManager"
        ),
        pytest.param("from langchain_core.runnables import Runnable", id="Runnable"),
        pytest.param(
            "from langchain_core.language_models import BaseChatModel",
            id="BaseChatModel",
        ),
        pytest.param(
            "from langchain_core.prompts import ChatPromptTemplate",
            id="ChatPromptTemplate",
        ),
        pytest.param("from langchain_core.documents import Document", id="Document"),
        pytest.param(
            "from langchain_core.vectorstores import InMemoryVectorStore",
            id="InMemoryVectorStore",
        ),
        pytest.param(
            "from langchain_core.runnables import RunnableLambda",
            id="RunnableLambda",
        ),
        pytest.param(
            "from langchain_core.tracers import LangChainTracer",
            id="LangChainTracer",
        ),
        pytest.param(
            "from langchain_core.output_parsers import PydanticOutputParser",
            id="PydanticOutputParser",
        ),
        pytest.param(
            "from langchain_core.rate_limiters import InMemoryRateLimiter",
            id="InMemoryRateLimiter",
        ),
    ],
)
@pytest.mark.benchmark
def test_import_time(benchmark: BenchmarkFixture, import_path: str) -> None:
    @benchmark
    def import_in_subprocess() -> None:
        subprocess.run([sys.executable, "-c", import_path], check=False)
