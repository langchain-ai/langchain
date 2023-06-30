import os
from pathlib import Path

from langchain.chains.openai_functions.openapi import get_openapi_chain


def test_openai_opeanapi() -> None:
    chain = get_openapi_chain(
        "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
    )
    output = chain.run("What are some options for a men's large blue button down shirt")

    assert isinstance(output, dict)


def test_openai_opeanapi_headers() -> None:
    BRANDFETCH_API_KEY = os.environ.get("BRANDFETCH_API_KEY")
    headers = {"Authorization": f"Bearer {BRANDFETCH_API_KEY}"}
    file_path = str(
        Path(__file__).parents[2] / "examples/brandfetch-brandfetch-2.0.0-resolved.json"
    )
    chain = get_openapi_chain(file_path, headers=headers)
    output = chain.run("I want to know about nike.comgg")

    assert isinstance(output, str)
