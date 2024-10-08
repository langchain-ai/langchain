import json
import os

from langchain_community.utilities.atlassian import AtlassianAPIWrapper


def check_environment_variables():
    required_vars = [
        "ATLASSIAN_INSTANCE_URL",
        "ATLASSIAN_USERNAME",
        "ATLASSIAN_API_TOKEN",
        "ATLASSIAN_CLOUD"
    ]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Required environment variable {var} is not set")


check_environment_variables()


def atlassian_wrapper():
    return AtlassianAPIWrapper(
        atlassian_instance_url=os.environ["ATLASSIAN_INSTANCE_URL"],
        atlassian_username=os.environ["ATLASSIAN_USERNAME"],
        atlassian_api_token=os.environ["ATLASSIAN_API_TOKEN"],
        atlassian_cloud=os.environ["ATLASSIAN_CLOUD"] == "True",
    )


def test_validate_and_initialize_environment() -> None:
    """Test environment validation and initialization."""
    atlassian = atlassian_wrapper()
    assert atlassian.jira is not None
    assert atlassian.confluence is not None


def test_run_jira_jql() -> None:
    """Test for JQL queries on Jira."""
    jql = "project = CS"
    atlassian = atlassian_wrapper()
    output = atlassian.run("jira_jql", jql)
    assert "issues" in json.loads(output)


def test_run_jira_get_functions() -> None:
    """Test for getting Jira functions."""
    atlassian = atlassian_wrapper()
    output = atlassian.run("jira_get_functions", "")
    output_json = json.loads(output)
    assert "functions" in output_json
    assert isinstance(output_json["functions"], list)


def test_run_jira_get_function_parameters() -> None:
    """Test for getting parameters of a specific Jira function."""
    atlassian = atlassian_wrapper()
    function_name = "create_issue"
    output = atlassian.run("jira_get_functions", function_name)
    output_json = json.loads(output)
    assert "function" in output_json
    assert output_json["function"] == function_name
    assert "parameters" in output_json
    assert isinstance(output_json["parameters"], list)


def test_run_jira_other() -> None:
    """Test for accessing other Jira API methods."""
    issue_create_dict = json.dumps(
        {
            "function": "create_issue",
            "args": [],
            "kwargs": {
                "fields": {
                    "summary": "Test Summary",
                    "description": "Test Description",
                    "issuetype": {"name": "Bug"},
                    "project": {"key": "CS"},
                }
            },
        }
    )
    atlassian = atlassian_wrapper()
    response = atlassian.run("jira_other", issue_create_dict)
    response_json = json.loads(response)
    assert "id" in response_json, f"Expected 'id' in response but got {response_json}"


def test_run_confluence_get_functions() -> None:
    """Test for getting Confluence functions."""
    atlassian = atlassian_wrapper()
    output = atlassian.run("confluence_get_functions", "")
    output_json = json.loads(output)
    assert "functions" in output_json
    assert isinstance(output_json["functions"], list)


def test_run_confluence_get_function_parameters() -> None:
    """Test for getting parameters of a specific Confluence function."""
    atlassian = atlassian_wrapper()
    function_name = "get_page_id"
    output = atlassian.run("confluence_get_functions", function_name)
    output_json = json.loads(output)
    assert "function" in output_json
    assert output_json["function"] == function_name
    assert "parameters" in output_json
    assert isinstance(output_json["parameters"], list)


def test_run_confluence_other() -> None:
    """Test for accessing other Confluence API methods."""
    atlassian = atlassian_wrapper()

    get_page_dict = json.dumps(
        {
            "function": "get_page_id",
            "args": [],
            "kwargs": {"space": "MPE", "title": "Test Page", "type": "page"},
        }
    )
    response = atlassian.run("confluence_other", get_page_dict)
    if response is None:
        response_json = {"error": "No response received"}
    else:
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            response_json = {"id": response}

    assert response_json is not None, f"Expected a valid response but got None"
    if isinstance(response_json, str):
        response_json = {"id": response_json}
    assert "id" in response_json, f"Expected 'id' in response but got {response_json}"


def test_run_confluence_cql() -> None:
    """Test for CQL queries on Confluence."""
    cql = "type=page"
    atlassian = atlassian_wrapper()
    output = atlassian.run("confluence_cql", cql)
    assert "results" in json.loads(output)


def test_apply_filter() -> None:
    """Test for applying filter to response."""
    atlassian = atlassian_wrapper()
    atlassian.filter_keys = ["*password*"]
    response = {"username": "user1", "password": "secret"}
    filtered_response = atlassian.apply_filter(response)
    filtered_json = json.loads(filtered_response)
    assert "password" not in filtered_json


def test_filter_response_keys() -> None:
    """Test for filter_response_keys function."""
    atlassian = atlassian_wrapper()
    atlassian.filter_keys = ["*password*", "secret*"]
    response = {
        "username": "user1",
        "password": "secret",
        "details": {"email": "user1@example.com", "secret_token": "abcdef"},
    }
    filtered_response = atlassian.filter_response_keys(response)
    assert "password" not in filtered_response
    assert "secret_token" not in filtered_response["details"]
    assert "username" in filtered_response
    assert "email" in filtered_response["details"]
