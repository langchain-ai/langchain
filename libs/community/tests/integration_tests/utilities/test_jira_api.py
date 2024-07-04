"""Integration test for JIRA API Wrapper."""

from langchain_community.utilities.jira import JiraAPIWrapper


def test_search() -> None:
    """Test for Searching issues on JIRA"""
    jql = "project = TP"
    jira = JiraAPIWrapper()  # type: ignore[call-arg]
    output = jira.run("jql", jql)
    assert "issues" in output


def test_getprojects() -> None:
    """Test for getting projects on JIRA"""
    jira = JiraAPIWrapper()  # type: ignore[call-arg]
    output = jira.run("get_projects", "")
    assert "projects" in output


def test_create_ticket() -> None:
    """Test the Create Ticket Call that Creates a Issue/Ticket on JIRA."""
    issue_string = (
        '{"summary": "Test Summary", "description": "Test Description",'
        ' "issuetype": {"name": "Bug"}, "project": {"key": "TP"}}'
    )
    jira = JiraAPIWrapper()  # type: ignore[call-arg]
    output = jira.run("create_issue", issue_string)
    assert "id" in output
    assert "key" in output


def test_create_confluence_page() -> None:
    """Test for getting projects on JIRA"""
    jira = JiraAPIWrapper()  # type: ignore[call-arg]
    create_page_dict = (
        '{"space": "ROC", "title":"This is the title",'
        '"body":"This is the body. You can use '
        '<strong>HTML tags</strong>!"}'
    )

    output = jira.run("create_page", create_page_dict)
    assert "type" in output
    assert "page" in output


def test_other() -> None:
    """Non-exhaustive test for accessing other JIRA API methods"""
    jira = JiraAPIWrapper()  # type: ignore[call-arg]
    issue_create_dict = """
        {
            "function":"issue_create",
            "kwargs": {
                "fields": {
                    "summary": "Test Summary",
                    "description": "Test Description",
                    "issuetype": {"name": "Bug"},
                    "project": {"key": "TP"}
                }
            }
        }
    """
    output = jira.run("other", issue_create_dict)
    assert "id" in output
    assert "key" in output
