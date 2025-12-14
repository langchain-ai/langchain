"""Benchmark tests for import time.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.parametrize(
    "import_path",
    [
        pytest.param(
            "from langchain_maritaca import ChatMaritaca", id="ChatMaritaca"
        ),
    ],
)
@pytest.mark.benchmark
def test_import_time(benchmark: BenchmarkFixture, import_path: str) -> None:
    """Benchmark import time for langchain-maritaca."""

    @benchmark  # type: ignore[misc]
    def import_in_subprocess() -> None:
        subprocess.run([sys.executable, "-c", import_path], check=True)
