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
