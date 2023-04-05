"""Test the APIOperation class."""


import os
from pathlib import Path
import subprocess
import tempfile

import pytest
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


@pytest.fixture
def robot_spec() -> OpenAPISpec:
    """Load the cached openapi spec for the robot mock server."""
    robot_spec_path = Path(__file__).parent / "test_specs" / "robot_openapi.yaml"
    return OpenAPISpec.from_file(robot_spec_path)


# Requires npm install -g typescript
def _check_is_valid_ts(namespace_def: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as ts_file:
        ts_file.write(namespace_def.encode())
        ts_file_name = ts_file.name

    try:
        # Compile the TypeScript file
        result = subprocess.run(["tsc", ts_file_name], capture_output=True, text=True)

        if result.returncode != 0:
            raise ValueError(
                f"TypeScript compilation failed:\n{result.stderr}\n\n{namespace_def}"
            )
    finally:
        os.remove(ts_file_name)


_ROBOT_METHODS = [
    ("/ask_for_help", "post"),
    ("/ask_for_passphrase", "get"),
    ("/get_state", "get"),
    # ("/goto/{x}/{y}/{z}", "post"), # Private type definitions required.
    ("/recycle", "delete"),
    ("/walk", "post"),
]


@pytest.mark.parametrize("path, method", _ROBOT_METHODS)
def test_parse_api_operations(robot_spec: OpenAPISpec, path: str, method: str):
    """Test the APIOperation class."""
    api_operation = APIOperation.from_openapi_spec(robot_spec, path, method)
    ts = api_operation.to_typescript()
    _check_is_valid_ts(ts)
