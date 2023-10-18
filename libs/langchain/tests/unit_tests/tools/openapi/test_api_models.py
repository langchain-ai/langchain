"""Test the APIOperation class."""
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

# Keep at top of file to ensure that pydantic test can be skipped before
# pydantic v1 related imports are attempted by openapi_pydantic.
from langchain.pydantic_v1 import _PYDANTIC_MAJOR_VERSION

if _PYDANTIC_MAJOR_VERSION != 1:
    pytest.skip(
        f"Pydantic major version {_PYDANTIC_MAJOR_VERSION} is not supported.",
        allow_module_level=True,
    )

import pytest
import yaml

from langchain.tools.openapi.utils.api_models import (
    APIOperation,
    APIRequestBody,
    APIRequestBodyProperty,
)
from langchain.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec

SPECS_DIR = Path(__file__).parents[2] / "examples" / "test_specs"


def _get_test_specs() -> Iterable[Path]:
    """Walk the test_specs directory and collect all files with the name 'apispec'
    in them.
    """
    if not SPECS_DIR.exists():
        raise ValueError
    return (
        Path(root) / file
        for root, _, files in os.walk(SPECS_DIR)
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


@pytest.mark.requires("openapi_pydantic")
def test_parse_api_operations() -> None:
    """Test the APIOperation class."""
    for spec_name, spec, path, method in http_paths_and_methods():
        try:
            APIOperation.from_openapi_spec(spec, path, method)
        except Exception as e:
            raise AssertionError(f"Error processing {spec_name}: {e} ") from e


@pytest.mark.requires("openapi_pydantic")
@pytest.fixture
def raw_spec() -> OpenAPISpec:
    """Return a raw OpenAPI spec."""
    from openapi_pydantic import Info

    return OpenAPISpec(
        info=Info(title="Test API", version="1.0.0"),
    )


@pytest.mark.requires("openapi_pydantic")
def test_api_request_body_from_request_body_with_ref(raw_spec: OpenAPISpec) -> None:
    """Test instantiating APIRequestBody from RequestBody with a reference."""
    from openapi_pydantic import (
        Components,
        MediaType,
        Reference,
        RequestBody,
        Schema,
    )

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


@pytest.mark.requires("openapi_pydantic")
def test_api_request_body_from_request_body_with_schema(raw_spec: OpenAPISpec) -> None:
    """Test instantiating APIRequestBody from RequestBody with a schema."""
    from openapi_pydantic import (
        MediaType,
        RequestBody,
        Schema,
    )

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


@pytest.mark.requires("openapi_pydantic")
def test_api_request_body_property_from_schema(raw_spec: OpenAPISpec) -> None:
    from openapi_pydantic import (
        Components,
        Reference,
        Schema,
    )

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
