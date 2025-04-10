import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore


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
            id="PromChatPromptTemplateptTemplate",
        ),
        pytest.param("from langchain_core.documents import Document", id="Document"),
        pytest.param(
            "from langchain_core.vectorstores import InMemoryVectorStore",
            id="InMemoryVectorStore",
        ),
    ],
)
@pytest.mark.benchmark
def test_import_time(benchmark: BenchmarkFixture, import_path: str) -> None:
    @benchmark
    def import_in_subprocess() -> None:
        subprocess.run([sys.executable, "-c", import_path], check=False)
