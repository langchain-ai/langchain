ATLASSIAN_JIRA_JQL_PROMPT = """
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

ATLASSIAN_CONFLUENCE_CQL_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Confluence cql API, useful when you need to search for 
    Confluence content. The input to this tool is a CQL query string, and will be passed into atlassian-python-api's 
    Confluence `cql` function.

    CQL (Confluence Query Language) allows you to create complex queries to search for content in Confluence.
    A well-constructed CQL query helps to avoid problems during the search.

    Here are some guidelines for building correct CQL queries:

    1. Use Quotes for Strings: Enclose string values in double quotes to avoid syntax errors.
       For example, use space = "TEST" instead of space = TEST.

    2. Escape Special Characters: If your query includes special characters (e.g., quotes, parentheses), escape them 
    with a backslash.
       For example, use title ~ "test\\ page" if the title contains special characters.

    3. Use Correct Field Names: Ensure you use the correct field names. Common fields include:
       - title
       - creator
       - contributor
       - space
       - type
       - created
       - modified

    4. Combine Conditions with Logical Operators: Use AND, OR, and NOT to combine multiple conditions.
       For example: title = "Test Page" AND creator = "user1" AND type = "page"

    5. Use Functions for Dynamic Values: Use Confluence functions like currentUser() for dynamic values.
       For example: creator = currentUser()

    6. Order and Limit Results: Use ORDER BY to sort results and LIMIT to restrict the number of results.
       For example: title = "Test Page" ORDER BY created DESC

    7. Avoid Invalid Characters: Do not use characters like backticks (`), single quotes ('), or other special 
    characters that are not part of the CQL syntax.

    Example Queries:
    - To find all the pages in space "DEMO" created by the current user, use:
      space = "DEMO" AND creator = currentUser() AND type = "page"
    - To find pages with titles that contain the word "test", use:
      title ~ "test"
    - To find all pages created by "user1" in the "DEMO" space, use:
      space = "DEMO" AND creator = "user1" AND type = "page"
    - To find pages created in the last 7 days, use:
      created >= -7d

    Common Errors and Fixes:
    - Error: space = DEMO
      Fix: space = "DEMO"
    - Error: title ~ test page
      Fix: title ~ "test page"
    - Error: space = `DEMO`
      Fix: space = "DEMO"
    - Error: creator = Daniel Gines AND type = page
      Fix: creator = "Daniel Gines" AND type = "page"
    - Error: `creator = "Daniel Gines" AND type = "page"`
      Fix: creator = "Daniel Gines" AND type = "page"

    For more advanced usage and functions, refer to the official Confluence CQL documentation:
    https://developer.atlassian.com/cloud/confluence/advanced-searching-using-cql/
    """


ATLASSIAN_JIRA_CATCH_ALL_PROMPT = """
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

ATLASSIAN_CONFLUENCE_CATCH_ALL_PROMPT = """
    This tool is a wrapper around atlassian-python-api's Confluence API.
    Use this tool if you need to perform any actions allowed by the atlassian-python-api Confluence API that are not 
    covered by other dedicated tools.
    The input to this tool is a dictionary specifying a function from atlassian-python-api's Confluence API, 
    as well as a list of arguments and dictionary of keyword arguments to pass into the function.
    For example, to get all the pages in a space, you would pass in the following dictionary:
    {{"function": "get_all_pages_from_space", "args": ["DEMO"], "kwargs": {{"limit":100}} }}
    For more information on the Confluence API, refer to https://atlassian-python-api.readthedocs.io/confluence.html
    """

ATLASSIAN_JIRA_GET_FUNCTIONS_PROMPT = """
    This tool retrieves a list of all available functions from the atlassian-python-api's Jira API. When called without 
    any input arguments, it returns a list of all available function names as strings, which can be called dynamically 
    via the wrapper. For example, the output could be `['search_issues', 'get_project', 'create_issue', ...]`. When 
    called with the name of a function as a query argument, it will return the parameters that the function accepts.
    For example, calling it with the query 'get_project' could return `['self', 'key']`.
    """

ATLASSIAN_CONFLUENCE_GET_FUNCTIONS_PROMPT = """
    This tool retrieves a list of all available functions from the atlassian-python-api's Confluence API. When called 
    without any input arguments, it returns a list of all available function names as strings, which can be called 
    dynamically via the wrapper. For example, the output could be `['get_page', 'create_page', 'delete_page', ...]`. 
    When called with the name of a function as a query argument, it will return the parameters that the function 
    accepts. For example, calling it with the query 'get_page' could return `['self', 'page_id']`.
    """
