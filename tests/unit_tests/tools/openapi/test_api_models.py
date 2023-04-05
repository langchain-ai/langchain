"""Test the APIOperation class."""


import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager

import pytest

from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


@pytest.fixture
def robot_spec() -> OpenAPISpec:
    """Load the cached openapi spec for the robot mock server."""
    robot_spec_path = Path(__file__).parent / "test_specs" / "robot_openapi.yaml"
    return OpenAPISpec.from_file(robot_spec_path)


@contextmanager
def _named_temp_file(suffix: str, content: str) -> ContextManager[str]:
    """Create a temporary file and yield its name."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(content.encode())
        temp_file_name = temp_file.name

    try:
        yield temp_file_name
    finally:
        os.remove(temp_file_name)


# Requires npm install -g typescript
def _compile_typescript(ts_file_name: str) -> subprocess.CompletedProcess:
    """Compile a typescript file using the tsc CLI post."""
    return subprocess.run(["tsc", ts_file_name], capture_output=True, text=True)


def _check_is_valid_ts(namespace_def: str) -> None:
    """Check that the typescript file is valid."""
    with _named_temp_file(".ts", namespace_def) as ts_file_name:
        result = _compile_typescript(ts_file_name)

        if result.returncode != 0:
            raise ValueError(
                f"TypeScript compilation failed:\n{result.stderr}\n\n{namespace_def}"
            )


_ROBOT_METHODS = [
    ("/ask_for_help", "post"),
    ("/ask_for_passphrase", "get"),
    ("/get_state", "get"),
    ("/goto/{x}/{y}/{z}", "post"),  # Private type definitions required.
    ("/recycle", "delete"),
    ("/walk", "post"),
]


@pytest.mark.parametrize("path, method", _ROBOT_METHODS)
def test_parse_api_operations(robot_spec: OpenAPISpec, path: str, method: str) -> None:
    """Test the APIOperation class."""
    api_operation = APIOperation.from_openapi_spec(robot_spec, path, method)
    ts = api_operation.to_typescript()
    _check_is_valid_ts(ts)
