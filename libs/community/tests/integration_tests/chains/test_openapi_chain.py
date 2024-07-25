from typing import Any, Dict, Union

import pytest
import requests
import yaml
from langchain_core.messages import AIMessage

from langchain_community.chains.openapi.chain import create_openapi_endpoint_chain


def _get_schema(response_json: Union[dict, list]) -> dict:
    if isinstance(response_json, list):
        response_json = response_json[0] if response_json else {}
    return {key: type(value).__name__ for key, value in response_json.items()}


def _get_api_spec() -> str:
    base_url = "https://jsonplaceholder.typicode.com"
    endpoints = [
        "/posts",
        "/comments",
    ]
    common_query_parameters = [
        {
            "name": "_limit",
            "in": "query",
            "required": False,
            "schema": {"type": "integer", "example": 2},
            "description": "Limit the number of results",
        }
    ]
    openapi_spec: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "JSONPlaceholder API", "version": "1.0.0"},
        "servers": [{"url": base_url}],
        "paths": {},
    }
    # Iterate over the endpoints to construct the paths
    for endpoint in endpoints:
        response = requests.get(base_url + endpoint)
        if response.status_code == 200:
            schema = _get_schema(response.json())
            openapi_spec["paths"][endpoint] = {
                "get": {
                    "summary": f"Get {endpoint[1:]}",
                    "parameters": common_query_parameters,
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": schema}
                                }
                            },
                        }
                    },
                }
            }
    return yaml.dump(openapi_spec, sort_keys=False)


@pytest.mark.requires("langchain_openai")
def test_create_openapi_endpoint_chain() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    api_spec = _get_api_spec()

    chain = create_openapi_endpoint_chain(llm, api_spec, allow_dangerous_requests=True)

    result = chain.invoke("What are the titles of the top two posts?")
    assert isinstance(result, AIMessage)
    assert "sunt aut facere" in result.content and "qui est esse" in result.content
