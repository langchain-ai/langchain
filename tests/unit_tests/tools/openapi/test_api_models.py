"""Test the APIOperation class."""
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest
import yaml

from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec

_DIR = Path(__file__).parent


def _get_test_specs() -> Iterable[Path]:
    """Walk the test_specs directory and collect all files with the name 'apispec'
    in them.
    """
    test_specs_dir = _DIR / "test_specs"
    return (
        Path(root) / file
        for root, _, files in os.walk(test_specs_dir)
        for file in files
        if file.startswith("apispec")
    )


def _get_paths_and_methods_from_spec_dictionary(
    spec: dict,
) -> Iterable[Tuple[str, str]]:
    """Return a tuple (paths, methods) for every path in spec."""
    valid_methods = [verb.value for verb in HTTPVerb]
    for path_name, path_item in spec["paths"].items():
        for method in valid_methods:
            if method in path_item:
                yield (path_name, method)


def http_paths_and_methods() -> List[Tuple[str, OpenAPISpec, str, str]]:
    """Return a args for every method in cached OpenAPI spec in test_specs."""
    http_paths_and_methods = []
    for test_spec in _get_test_specs():
        spec_name = test_spec.parent.name
        if test_spec.suffix == ".json":
            with test_spec.open("r") as f:
                spec = json.load(f)
        else:
            with test_spec.open("r") as f:
                spec = yaml.safe_load(f.read())
        parsed_spec = OpenAPISpec.from_file(test_spec)
        for path, method in _get_paths_and_methods_from_spec_dictionary(spec):
            http_paths_and_methods.append(
                (
                    spec_name,
                    parsed_spec,
                    path,
                    method,
                )
            )
    return http_paths_and_methods


@pytest.mark.parametrize(
    "spec_name, spec, path, method",
    http_paths_and_methods(),
)
def test_parse_api_operations(
    spec_name: str, spec: OpenAPISpec, path: str, method: str
) -> None:
    """Test the APIOperation class."""
    try:
        APIOperation.from_openapi_spec(spec, path, method)
    except Exception as e:
        raise AssertionError(f"Error processong {spec_name}: {e} ") from e
