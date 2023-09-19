import json

import pytest
import responses

from langchain.utilities.graphql import GraphQLAPIWrapper

TEST_ENDPOINT = "http://testserver/graphql"

# Mock GraphQL response for testing
MOCK_RESPONSE = {
    "data": {
        "allUsers": [{"name": "Alice"}],
        "__schema": {
            "queryType": {"name": "Query"},
            "types": [
                {
                    "kind": "OBJECT",
                    "name": "Query",
                    "fields": [
                        {
                            "name": "allUsers",
                            "args": [],
                            "type": {
                                "kind": "NON_NULL",
                                "name": None,
                                "ofType": {
                                    "kind": "OBJECT",
                                    "name": "allUsers",
                                    "ofType": None,
                                },
                            },
                        }
                    ],
                    "inputFields": None,
                    "interfaces": [],
                    "enumValues": None,
                    "possibleTypes": None,
                },
                {
                    "kind": "SCALAR",
                    "name": "String",
                },
                {
                    "kind": "OBJECT",
                    "name": "allUsers",
                    "description": None,
                    "fields": [
                        {
                            "name": "name",
                            "description": None,
                            "args": [],
                            "type": {
                                "kind": "NON_NULL",
                                "name": None,
                                "ofType": {
                                    "kind": "SCALAR",
                                    "name": "String",
                                    "ofType": None,
                                },
                            },
                        },
                    ],
                    "inputFields": None,
                    "interfaces": [],
                    "enumValues": None,
                    "possibleTypes": None,
                },
                {
                    "kind": "SCALAR",
                    "name": "Boolean",
                },
            ],
        },
    }
}


@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run() -> None:
    responses.add(responses.POST, TEST_ENDPOINT, json=MOCK_RESPONSE, status=200)

    query = "query { allUsers { name } }"
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT,
        custom_headers={"Authorization": "Bearer testtoken"},
    )
    result = graphql_wrapper.run(query)

    expected_result = json.dumps(MOCK_RESPONSE["data"], indent=2)
    assert result == expected_result
