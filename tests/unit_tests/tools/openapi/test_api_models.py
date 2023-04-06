"""Test the APIOperation class."""
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest
import yaml
from openapi_schema_pydantic import (
    Components,
    Info,
    MediaType,
    Reference,
    RequestBody,
    Schema,
)

from langchain.tools.openapi.utils.api_models import (
    APIOperation,
    APIRequestBody,
    APIRequestBodyProperty,
)
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


@pytest.fixture
def raw_spec() -> OpenAPISpec:
    """Return a raw OpenAPI spec."""
    return OpenAPISpec(
        info=Info(title="Test API", version="1.0.0"),
    )


def test_api_request_body_from_request_body_with_ref(raw_spec: OpenAPISpec) -> None:
    """Test instantiating APIRequestBody from RequestBody with a reference."""
    raw_spec.components = Components(
        schemas={
            "Foo": Schema(
                type="object",
                properties={
                    "foo": Schema(type="string"),
                    "bar": Schema(type="number"),
                },
                required=["foo"],
            )
        }
    )
    media_type = MediaType(
        schema=Reference(
            ref="#/components/schemas/Foo",
        )
    )
    request_body = RequestBody(content={"application/json": media_type})
    api_request_body = APIRequestBody.from_request_body(request_body, raw_spec)
    assert api_request_body.description is None
    assert len(api_request_body.properties) == 2
    foo_prop = api_request_body.properties[0]
    assert foo_prop.name == "foo"
    assert foo_prop.required is True
    bar_prop = api_request_body.properties[1]
    assert bar_prop.name == "bar"
    assert bar_prop.required is False
    assert api_request_body.media_type == "application/json"


def test_api_request_body_from_request_body_with_schema(raw_spec: OpenAPISpec) -> None:
    """Test instantiating APIRequestBody from RequestBody with a schema."""
    request_body = RequestBody(
        content={
            "application/json": MediaType(
                schema=Schema(type="object", properties={"foo": Schema(type="string")})
            )
        }
    )
    api_request_body = APIRequestBody.from_request_body(request_body, raw_spec)
    assert api_request_body.properties == [
        APIRequestBodyProperty(
            name="foo",
            required=False,
            type="string",
            default=None,
            description=None,
            properties=[],
            references_used=[],
        )
    ]
    assert api_request_body.media_type == "application/json"


def test_api_request_body_property_from_schema(raw_spec: OpenAPISpec) -> None:
    raw_spec.components = Components(
        schemas={
            "Bar": Schema(
                type="number",
            )
        }
    )
    schema = Schema(
        type="object",
        properties={
            "foo": Schema(type="string"),
            "bar": Reference(ref="#/components/schemas/Bar"),
        },
        required=["bar"],
    )
    api_request_body_property = APIRequestBodyProperty.from_schema(
        schema, "test", required=True, spec=raw_spec
    )
    expected_sub_properties = [
        APIRequestBodyProperty(
            name="foo",
            required=False,
            type="string",
            default=None,
            description=None,
            properties=[],
            references_used=[],
        ),
        APIRequestBodyProperty(
            name="bar",
            required=True,
            type="number",
            default=None,
            description=None,
            properties=[],
            references_used=["Bar"],
        ),
    ]
    assert api_request_body_property.properties[0] == expected_sub_properties[0]
    assert api_request_body_property.properties[1] == expected_sub_properties[1]
    assert api_request_body_property.type == "object"
    assert api_request_body_property.properties[1].references_used == ["Bar"]
