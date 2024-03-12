from pathlib import Path

import pytest
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn

from langchain_community.utilities.openapi import (  # noqa: E402 # ignore: community-import
    OpenAPISpec,
)

EXPECTED_OPENAI_FUNCTIONS_HEADER_PARAM = [
    {
        "name": "showPetById",
        "description": "Info for a specific pet",
        "parameters": {
            "type": "object",
            "properties": {
                "headers": {
                    "type": "object",
                    "properties": {
                        "header_param": {
                            "type": "string",
                            "description": "A header param",
                        }
                    },
                    "required": ["header_param"],
                }
            },
        },
    }
]


@pytest.mark.requires("openapi_pydantic")
def test_header_param() -> None:
    spec = OpenAPISpec.from_file(
        Path(__file__).parent.parent
        / "data"
        / "openapi_specs"
        / "openapi_spec_header_param.json",
    )

    openai_functions, _ = openapi_spec_to_openai_fn(spec)

    assert openai_functions == EXPECTED_OPENAI_FUNCTIONS_HEADER_PARAM


EXPECTED_OPENAI_FUNCTIONS_NESTED_REF = [
    {
        "name": "addPet",
        "description": "Add a new pet to the store",
        "parameters": {
            "type": "object",
            "properties": {
                "json": {
                    "properties": {
                        "id": {
                            "type": "integer",
                            "schema_format": "int64",
                            "example": 10,
                        },
                        "name": {"type": "string", "example": "doggie"},
                        "tags": {
                            "items": {
                                "properties": {
                                    "id": {"type": "integer", "schema_format": "int64"},
                                    "model_type": {"type": "number"},
                                },
                                "type": "object",
                            },
                            "type": "array",
                        },
                    },
                    "type": "object",
                    "required": ["name"],
                }
            },
        },
    }
]


@pytest.mark.requires("openapi_pydantic")
def test_nested_ref_in_openapi_spec() -> None:
    spec = OpenAPISpec.from_file(
        Path(__file__).parent.parent
        / "data"
        / "openapi_specs"
        / "openapi_spec_nested_ref.json",
    )

    openai_functions, _ = openapi_spec_to_openai_fn(spec)

    assert openai_functions == EXPECTED_OPENAI_FUNCTIONS_NESTED_REF
