import os

from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit

from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.utilities.jira import JiraAPIWrapper

os.environ["JIRA_API_TOKEN"] = "ATATT3xFfGF0d4A0QKjxYxM4vvu9DRjIl7N04456LlK7NizKbWHmkkDPV70ofC-3xZaG5g4z6nKeKu_u5n5jQ-TNxUf63TmZyn4DaQtrcCiUo4IuzaCTMJGRhYTN-hG79ZqTg2TyRrDzNHP6AgHNj3SzAvU4ex4pfEd_W1uqvlbm6ZaKAl297LU=BFFCFA45"

os.environ["JIRA_USERNAME"] = "asankar@clickup.com"

os.environ["JIRA_INSTANCE_URL"] = "https://testing-clickup.atlassian.net/"

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(JiraAPIWrapper())
llm = OpenAI(temperature=0, openai_api_key="")

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("Can you create a new issue called Test Issue 2 in the project My Kanban Project with key KAN?")
agent.run("Can you assign the issue Test Issue 2 to to asankar@clickup.com")


from atlassian import Confluence, Jira

# Define your Jira instance URL and credentials
jira_url = 'https://testing-clickup.atlassian.net/'
jira_username = 'asankar@clickup.com'
jira_api_token = ""

# Create a Jira client
# jira = JIRA(server=jira_url, basic_auth=(username, password))

jira_client = Jira(
    url=jira_url,
    username=jira_username,
    password=jira_api_token,
    cloud=True,
)

project = 'KAN'
summary = 'Test Issue 2'

query = f'project = {project} AND summary = "{summary}"'

def parse_issues(issues):
    parsed = []
    for issue in issues["issues"]:
        key = issue["key"]
        summary = issue["fields"]["summary"]
        created = issue["fields"]["created"][0:10]
        priority = issue["fields"]["priority"]["name"]
        status = issue["fields"]["status"]["name"]
        try:
            assignee = issue["fields"]["assignee"]["displayName"]
        except Exception:
            assignee = "None"
        rel_issues = {}
        for related_issue in issue["fields"]["issuelinks"]:
            if "inwardIssue" in related_issue.keys():
                rel_type = related_issue["type"]["inward"]
                rel_key = related_issue["inwardIssue"]["key"]
                rel_summary = related_issue["inwardIssue"]["fields"]["summary"]
            if "outwardIssue" in related_issue.keys():
                rel_type = related_issue["type"]["outward"]
                rel_key = related_issue["outwardIssue"]["key"]
                rel_summary = related_issue["outwardIssue"]["fields"]["summary"]
            rel_issues = {"type": rel_type, "key": rel_key, "summary": rel_summary}
        parsed.append(
            {
                "key": key,
                "summary": summary,
                "created": created,
                "assignee": assignee,
                "priority": priority,
                "status": status,
                "related_issues": rel_issues,
            }
        )
    return parsed

issues = jira_client.jql(query)
# print(issues)
parsed_issues = parse_issues(issues)
print(parsed_issues)

# Get all projects
projects = jira_client.projects()
print(projects)

# Print project information
for project in projects:
    print("*****")
    print(project)
    print(f"Project Key: " + str(project["key"]))
    print(f"Project Name:  " + str(project["name"]))
    print(f"Project ID:  " + str(project["id"]))
    print(f"Project URL:  " + str(project["self"]))
    print("\n")


