import json

import pytest

from langchain.chains.openai_functions.openapi import get_openapi_chain

api_spec = {
    "openapi": "3.0.0",
    "info": {"title": "JSONPlaceholder API", "version": "1.0.0"},
    "servers": [{"url": "https://jsonplaceholder.typicode.com"}],
    "paths": {
        "/posts": {
            "get": {
                "summary": "Get posts",
                "parameters": [
                    {
                        "name": "_limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer", "example": 2},
                        "description": "Limit the number of results",
                    },
                ],
            }
        }
    },
}


@pytest.mark.requires("openapi_pydantic")
@pytest.mark.requires("langchain_openai")
def test_openai_openapi_chain() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = get_openapi_chain(json.dumps(api_spec), llm)
    output = chain.invoke({"query": "Fetch the top two posts."})
    assert len(output["response"]) == 2
