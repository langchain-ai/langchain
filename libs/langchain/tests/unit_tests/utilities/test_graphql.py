import json

import pytest
import responses


from responses import matchers
from langchain.utilities.graphql import GraphQLAPIWrapper

TEST_ENDPOINT1 = "http://testserver1/graphql"
TEST_ENDPOINT2 = "http://testserver2/graphql"

__SCHEMA = {
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
                        },
                        {
                            "name": "user",
                            "args": [
                                {
                                    "name": "name",
                                    "type": {
                                        "kind": "NON_NULL",
                                        "name": None,
                                        "ofType": {
                                            "kind": "SCALAR",
                                            "name": "String",
                                            "ofType": None,
                                        },
                                    },
                                }
                            ],
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
        }

# Mock GraphQL response for testing
MOCK_RESPONSE1 = {
    "data": {
        "allUsers": [{"name": "Alice"}, {"name": "John"}, {"name": "May"}],
        "__schema": __SCHEMA
    }
}

MOCK_RESPONSE2 = {
    "data": {
        "allUsers": [{"name": "Alice"}],
        "__schema": __SCHEMA
    }
}

@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run() -> None:
    responses.add(responses.POST, TEST_ENDPOINT1, json=MOCK_RESPONSE1, status=200)

    query = "query { allUsers { name } }"
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT1,
        custom_headers={"Authorization": "Bearer testtoken"},
    )
    result = graphql_wrapper.run(query)

    expected_result = json.dumps(MOCK_RESPONSE1["data"], indent=2)
    assert result == expected_result

@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run_with_query_variables() -> None:
    responses.add(responses.POST, TEST_ENDPOINT1, json=MOCK_RESPONSE2, status=200)
    query = "query($name: String!) { user(name: $name) { name } }"
    query_variables = {"name": "Alice"}

    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT1,
        custom_headers={"Authorization": "Bearer testtoken"},
    )

    result = graphql_wrapper.run(query, query_variables)

    expected_result = json.dumps(MOCK_RESPONSE2["data"], indent=2)
    assert result == expected_result

@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run_with_gql_endpoint() -> None:
    responses.post(
        url=TEST_ENDPOINT1,
        json=MOCK_RESPONSE1,
        status=200
    )

    responses.post(
        url=TEST_ENDPOINT2,
        json=MOCK_RESPONSE2,
        status=200
    )

    query = "query { allUsers { name } }"
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT1,
        custom_headers={"Authorization": "Bearer testtoken"},
    )
    result = graphql_wrapper.run(query, None, TEST_ENDPOINT2)

    expected_result = json.dumps(MOCK_RESPONSE2["data"], indent=2)
    assert result == expected_result

@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run_with_headers() -> None:

    testuser_auth = {"Authorization": "Bearer testtoken"}
    admin_auth = {"Authorization": "Bearer admintoken"}

    responses.post(
        url=TEST_ENDPOINT1,
        match=[matchers.header_matcher(testuser_auth)],
        json=MOCK_RESPONSE2,
        status=200
    )

    responses.post(
        url=TEST_ENDPOINT1,
        match=[matchers.header_matcher(admin_auth)],
        json=MOCK_RESPONSE1,
        status=200
    )

    query = "query { allUsers { name } }"
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT1,
        custom_headers=testuser_auth,
    )

    result = graphql_wrapper.run(query, None, None, admin_auth)

    expected_result = json.dumps(MOCK_RESPONSE1["data"], indent=2)
    assert result == expected_result

@pytest.mark.requires("gql", "requests_toolbelt")
@responses.activate
def test_run_with_gqlendpoint_n_headers() -> None:

    testuser_auth = {"Authorization": "Bearer testtoken"}
    admin_auth = {"Authorization": "Bearer admintoken"}

    responses.post(
        url=TEST_ENDPOINT1,
        match=[matchers.header_matcher(testuser_auth)],
        json=MOCK_RESPONSE2,
        status=200
    )

    responses.post(
        url=TEST_ENDPOINT2,
        match=[matchers.header_matcher(admin_auth)],
        json=MOCK_RESPONSE1,
        status=200
    )

    query = "query { allUsers { name } }"
    graphql_wrapper = GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT1,
        custom_headers=testuser_auth,
    )

    result = graphql_wrapper.run(query, None, TEST_ENDPOINT2, admin_auth)

    expected_result = json.dumps(MOCK_RESPONSE1["data"], indent=2)
    assert result == expected_result