import pytest
import json
from typing import Optional, Dict, Any
from langchain.utilities.graphql import GraphQLAPIWrapper
from langchain.tools.graphql.tool import BaseGraphQLTool

TEST_ENDPOINT1 = "http://testserver1/graphql"
TEST_ENDPOINT2 = "http://testserver2/graphql"

MOCK_RESPONSE1 = {
    "data": {
        "allUsers": [{"name": "John"}, {"name": "Wane"}, {"name": "Greg"}]
    }
}

MOCK_STR_RESPONSE1 = json.dumps(MOCK_RESPONSE1["data"], indent=2)

MOCK_RESPONSE2 = {
    "data": {
        "allUsers": [{"name": "Alice"}, {"name": "Nick"}]
    }
}

MOCK_STR_RESPONSE2 = json.dumps(MOCK_RESPONSE2["data"], indent=2)

MOCK_RESPONSE3 = {
    "data": {
        "allUsers": [{"name": "Alice"}]
    }
}

MOCK_STR_RESPONSE3 = json.dumps(MOCK_RESPONSE3["data"], indent=2)

MOCK_RESPONSE4 = {
    "data": {
        "allUsers": [{"name": "Admin"}]
    }
}

MOCK_STR_RESPONSE4 = json.dumps(MOCK_RESPONSE4["data"], indent=2)

class _MockGraphQLAPIWrapper(GraphQLAPIWrapper):
    def run(self, query: str, query_variables: Optional[Dict[str, Any]] = None, graphql_endpoint: str = None,
            headers: Optional[Dict[str, str]] = None) -> str:
        if query_variables is not None and query_variables["name"] is "Alice":
            return MOCK_STR_RESPONSE3
        elif graphql_endpoint is TEST_ENDPOINT2:
            return MOCK_STR_RESPONSE2
        elif headers is not None and headers["Authorization"] is "Bearer admintoken":
            return MOCK_STR_RESPONSE4
        else:
            return MOCK_STR_RESPONSE1


@pytest.fixture
def mock_graphqlapi_wrapper() -> GraphQLAPIWrapper:
    return _MockGraphQLAPIWrapper(graphql_endpoint=TEST_ENDPOINT1)


def test_run_with_query(mock_graphqlapi_wrapper: GraphQLAPIWrapper) -> None:
    query = "query { allUsers { name } }"
    expected_result = json.dumps(MOCK_STR_RESPONSE1, indent=2)
    tool = BaseGraphQLTool(graphql_wrapper=mock_graphqlapi_wrapper)
    result = tool.run(query)
    assert result == expected_result

def test_run_with_query_variables(mock_graphqlapi_wrapper: GraphQLAPIWrapper) -> None:
    tool_inputs = {"query": "query($name: String) { allUsers(name: $name) { name } }", "query_variables": {"name": "Alice"}}
    expected_result = json.dumps(MOCK_STR_RESPONSE3, indent=2)
    tool = BaseGraphQLTool(graphql_wrapper=mock_graphqlapi_wrapper)
    result = tool.run(tool_inputs)
    assert result == expected_result

def test_run_with_gql_endpoint(mock_graphqlapi_wrapper: GraphQLAPIWrapper) -> None:
    tool_inputs = {"query": "query { allUsers { name } }", "graphql_endpoint": TEST_ENDPOINT2}
    expected_result = json.dumps(MOCK_STR_RESPONSE2, indent=2)
    tool = BaseGraphQLTool(graphql_wrapper=mock_graphqlapi_wrapper)
    result = tool.run(tool_inputs)
    assert result == expected_result

def test_run_with_headers(mock_graphqlapi_wrapper: GraphQLAPIWrapper) -> None:
    tool_inputs = {"query": "query { allUsers { name } }", "headers": {"Authorization": "Bearer admintoken"}}
    expected_result = json.dumps(MOCK_STR_RESPONSE4, indent=2)
    tool = BaseGraphQLTool(graphql_wrapper=mock_graphqlapi_wrapper)
    result = tool.run(tool_inputs)
    assert result == expected_result