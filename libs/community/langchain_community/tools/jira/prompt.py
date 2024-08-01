# flake8: noqa
JIRA_ISSUE_CREATE_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira issue_create API, useful when you need to create a 
    Jira issue. The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed 
    into atlassian-python-api's Jira `issue_create` function. For example, to create a low priority task called 
    "test issue" with description "test description", you would pass in the following dictionary: 
    {{"summary": "test issue", "description": "test description", "issuetype": {{"name": "Task"}}, 
    "priority": {{"name": "Low"}}}}
    """

JIRA_GET_ALL_PROJECTS_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira project API, 
    useful when you need to fetch all the projects the user has access to, find out how many projects there are, 
    or as an intermediary step that involv searching by projects. 
    there is no input to this tool.
    """

JIRA_JQL_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira jql API, useful when you need to search for Jira issues.
    The input to this tool is a JQL query string, and will be passed into atlassian-python-api's Jira `jql` function.

    JQL (Jira Query Language) allows you to create complex queries to search for issues in Jira.
    A well-constructed JQL query helps to avoid problems during the search.

    Here are some guidelines for building correct JQL queries:

    1. Use Quotes for Strings: Enclose string values in double quotes to avoid syntax errors.
       For example, use project = "Test" instead of project = Test.

    2. Escape Special Characters: If your query includes special characters (e.g., quotes, parentheses), escape them 
    with a backslash.
       For example, use summary ~ "test\\ issue" if the summary contains special characters.

    3. Use Correct Field Names: Ensure you use the correct field names. Common fields include:
       - project
       - assignee
       - reporter
       - summary
       - status
       - priority
       - created
       - updated

    4. Combine Conditions with Logical Operators: Use AND, OR, and NOT to combine multiple conditions.
       For example: project = "Test" AND assignee = "user1" AND status = "Open"

    5. Use Functions for Dynamic Values: Use Jira functions like currentUser() for dynamic values.
       For example: assignee = currentUser()

    6. Order and Limit Results: Use ORDER BY to sort results and LIMIT to restrict the number of results.
       For example: project = "Test" ORDER BY created DESC

    7. Avoid Invalid Characters: Do not use characters like backticks (`), single quotes ('), or other special 
    characters that are not part of the JQL syntax.

    Example Queries:
    - To find all the issues in project "Test" assigned to the current user, use:
      project = "Test" AND assignee = currentUser()
    - To find issues with summaries that contain the word "test", use:
      summary ~ "test"
    - To find all open issues assigned to "user1" in the "Test" project, use:
      project = "Test" AND assignee = "user1" AND status = "Open"
    - To find issues created in the last 7 days, use:
      created >= -7d

    Common Errors and Fixes:
    - Error: project = Test
      Fix: project = "Test"
    - Error: summary ~ test issue
      Fix: summary ~ "test issue"
    - Error: project = `Test`
      Fix: project = "Test"
    - Error: assignee = Daniel Gines AND status = Open
      Fix: assignee = "Daniel Gines" AND status = "Open"
    - Error: `assignee = "Daniel Gines" AND status = "Open"`
      Fix: assignee = "Daniel Gines" AND status = "Open"

    For more advanced usage and functions, refer to the official Jira JQL documentation:
    https://support.atlassian.com/jira-software-cloud/docs/advanced-search-reference-jql-fields/
    """

JIRA_CATCH_ALL_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira API.
    There are other dedicated tools for fetching all projects, and creating and searching for issues, 
    use this tool if you need to perform any other actions allowed by the atlassian-python-api Jira API.
    The input to this tool is a dictionary specifying a function from atlassian-python-api's Jira API, 
    as well as a list of arguments and dictionary of keyword arguments to pass into the function.
    For example, to get all the users in a group, while increasing the max number of results to 100, you would
    pass in the following dictionary: {{"function": "get_all_users_from_group", "args": ["group"], 
    "kwargs": {{"limit":100}} }}
    or to find out how many projects are in the Jira instance, you would pass in the following string:
    {{"function": "projects"}}
    For more information on the Jira API, refer to https://atlassian-python-api.readthedocs.io/jira.html
    """

JIRA_CONFLUENCE_PAGE_CREATE_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Confluence atlassian-python-api API, useful when you need 
    to create a Confluence page. The input to this tool is a dictionary specifying the fields of the Confluence page, 
    and will be passed into atlassian-python-api's Confluence `create_page` function. For example, to create a page in 
    the DEMO space titled "This is the title" with body "This is the body. 
    You can use <strong>HTML tags</strong>!", you would pass in the following dictionary: 
    {{"space": "DEMO", "title":"This is the title","body":"This is the body. 
    You can use <strong>HTML tags</strong>!"}} 
    """

JIRA_TICKETS_FOR_USER_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira jql API, useful when you need to find out how many tickets 
    are assigned to a specific user. The input to this tool is a JQL query string specifying the user, and will be 
    passed into atlassian-python-api's Jira `jql` function. For example, to find all the issues assigned to the user 
    "john.doe", you would pass in the following string: assignee = john.doe
    """

JIRA_TICKETS_FOR_USER_IN_PROJECT_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira jql API, useful when you need to find out how many tickets 
    are assigned to a specific user within a specific project. The input to this tool is a JQL query string specifying 
    the user and the project, and will be passed into atlassian-python-api's Jira `jql` function. For example, 
    to find all the issues in project "TestProject" assigned to the user "john.doe", you would pass in the following 
    string: project = TestProject AND assignee = john.doe
    """

JIRA_GET_GROUPS_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira `get_groups` API, useful when you need to search for 
    Jira groups. The input to this tool is a dictionary specifying the query parameters for the group search, 
    and will be passed into atlassian-python-api's Jira `get_groups` function. For example, to search for groups with 
    names containing the word "admin" and excluding the group "jira-administrators", you would pass in the following 
    dictionary: {{"query": "admin", "exclude": "jira-administrators", "limit": 20}}
    The limit parameter is optional and defaults to 20 if not provided.
    """

JIRA_USERS_GET_ALL_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira users_get_all API, useful when you need to fetch 
    all users. The input to this tool is a dictionary specifying the start and limit of the users to retrieve.
    For example, to get the first 50 users, you would pass in the following dictionary: 
    {"start": 0, "limit": 50}
    """

JIRA_USER_FIND_BY_USER_STRING_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira user_find_by_user_string API, useful for fuzzy search 
    using display name, email address, or property, or an exact search for account ID or username.
    The input to this tool is a dictionary specifying the search parameters.
    For example, to search for a user by display name or email address, you would pass in the following dictionary:
    {"query": "John Doe"}
    To search for a user by account ID, you would pass in the following dictionary:
    {"account_id": "5b10a2844c20165700ede21g"}
    """

JIRA_PROJECT_LEADERS_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira project_leaders API, useful when you need to fetch the 
    project leaders for all projects. There is no input to this tool.
    """

JIRA_GET_ALL_PROJECT_ISSUES_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira get_all_project_issues API, useful when you need to fetch 
    all issues for a specific project. The input to this tool is a dictionary specifying the project key, fields to 
    retrieve, start index, and limit of the issues to retrieve. For example, to get all issues for project "TEST" with 
    all fields, starting from the first issue, you would pass in the following dictionary:
    {"project": "TEST", "fields": "*all", "start": 0, "limit": 50}
    """

JIRA_GET_PROJECT_VALIDATED_KEY_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira get_project_validated_key API, useful when you need to 
    validate a project key. The input to this tool is a dictionary specifying the project key.
    For example, to validate the project key "TEST", you would pass in the following dictionary:
    {"key": "TEST"}
    """

JIRA_GET_ALL_PROJECT_CATEGORIES_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira get_all_project_categories API, useful when you need to 
    fetch all project categories. There is no input to this tool.
    """

JIRA_GET_ALL_STATUSES_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira get_all_statuses API, useful when you need to fetch all 
    statuses. There is no input to this tool.
    """

JIRA_JQL_GET_LIST_OF_TICKETS_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira jql_get_list_of_tickets API, useful when you need to get 
    issues from a JQL search result with all related fields. The input to this tool is a dictionary specifying the JQL 
    query, fields to retrieve, start index, limit of the issues to retrieve, expand, and whether to validate the query.
    For example, to get issues from a JQL search result with all fields, starting from the first issue, you would pass 
    in the following dictionary: {"jql": "project = TEST", "fields": "*all", "start": 0, "limit": 50}
    """

JIRA_GET_ALL_USERS_FROM_GROUP_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Jira API for retrieving all users from a specific group.
    The input to this tool is a query string specifying the group and will be passed into the 
    `get_all_users_from_group` function.
    For example, to find all the users from the group "estabilis-team", you would pass in the following string:
    estabilis-team
    It also supports additional parameters to include inactive users and limit the number of results. 
    These can be provided as optional arguments: include_inactive_users (default is False) and limit (default is 100).
    """

JIRA_GET_FUNCTIONS_PROMPT = """
    This tool retrieves a list of all available functions from the atlassian-python-api's Jira API. When called without 
    any input arguments, it returns a list of all available function names as strings, which can be called dynamically 
    via the wrapper. For example, the output could be `['search_issues', 'get_project', 'create_issue', ...]`. When 
    called with the name of a function as a query argument, it will return the parameters that the function accepts.
    For example, calling it with the query 'get_project' could return `['self', 'key']`.
    """
