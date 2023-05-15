import json

import pytest
import responses

from langchain.utilities.graphql import GraphQLAPIWrapper

TEST_ENDPOINT = "http://testserver/graphql"

# Mock GraphQL response for testing
MOCK_RESPONSE = {
    "data": {"allUsers": [{"id": 1, "name": "Alice", "email": "alice@example.com"}]}
}


@pytest.fixture
def graphql_wrapper() -> GraphQLAPIWrapper:
    return GraphQLAPIWrapper(
        graphql_endpoint=TEST_ENDPOINT,
        custom_headers={"Authorization": "Bearer testtoken"},
    )


@responses.activate
def test_run(graphql_wrapper: GraphQLAPIWrapper) -> None:
    responses.add(responses.POST, TEST_ENDPOINT, json=MOCK_RESPONSE, status=200)

    query = "query { allUsers { id, name, email } }"
    result = graphql_wrapper.run(query)

    expected_result = json.dumps(MOCK_RESPONSE, indent=2)
    assert result == expected_result
