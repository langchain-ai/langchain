# flake8: noqa
JIRA_CREATE_PROMPT = ('This tool is a wrapper around atlassian-python-api\'s Jira issue create API. '
                      'Useful for when you need to create a Jira issue/task/ticket. '
                      'The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed into atlassian-python-api\'s Jira issue_create function, '
                      'For example, to create a low priority task called "test issue" with description "test description", you would pass in the following dictionary: '
                      '{{"summary": "test issue", "description": "test description", "issuetype": {{"name": "Task"}}, "priority": {{"name": "Low"}}}}')

JIRA_JQL_PROMPT = "This tool is a wrapper around atlassian-python-api's Jira jql API. " \
                  "Useful for when you need to search for Jira issues/tasks/tickets, or find out information about an issue. " \
                  "The input to this tool is a JQL query string, and will be passed into atlassian-python-api's Jira jql function, " \
                  "For example, find all the issues in project 'Test' assigned to the me, you would pass in the following string: " \
                  "'project = Test AND assignee = currentUser()'"
