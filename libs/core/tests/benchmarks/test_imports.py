import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]


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
    @benchmark  # type: ignore[misc]
    def import_in_subprocess() -> None:
        subprocess.run([sys.executable, "-c", import_path], check=True)
