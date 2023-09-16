import os

from langchain.agents.agent_toolkits.clickup.toolkit import ClickupToolkit

from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.utilities.clickup import ClickupAPIWrapper
import requests

oauth_client_id = "VR9ER2KI35L9SL4QWQD3VXK5UUUQBCD9"
ouath_client_secret = "O5E2F09WQG4BVDF283ZYLWBWT4OPJPNL0KDKKU08U3EJ3MH00FXVF1II9UOCS5KG"
code = "S695H3CSYBXDDZU8JKO6M79QNQES7N6G"

clickup = ClickupAPIWrapper(oauth_client_id=oauth_client_id, oauth_client_secret=ouath_client_secret, code=code)

toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup)
llm = OpenAI(temperature=0, openai_api_key="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")

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

print("Can you get all the teams that the user is authorized to access?")
agent.run("Can you get all the teams that the user is authorized to access?")

print("Can you get all the folders for the team?")
agent.run("Can you get all the folders for the team?")

print("Can you get all the spaces available to the team?")
agent.run("Can you get all the spaces available to the team?")

print("Can you get a task with id 86a0t44tq")
agent.run("Can you get a task with id 86a0t44tq")

print("Create a task")
agent.run("Can you create a task called Test Task with description My test task with high priority and status open?")

print("Create a list")
agent.run("Can you create a list called Test List with content My test list with high priority and status red?")


print("Create a folder")
agent.run("Can you create a folder called Test Folder")

"""
Testing the clickup API
"""


# import os
# import requests

# # Fill in your data values

# app_name="TestAuth"
# client_id = "VR9ER2KI35L9SL4QWQD3VXK5UUUQBCD9"
# client_secret = "O5E2F09WQG4BVDF283ZYLWBWT4OPJPNL0KDKKU08U3EJ3MH00FXVF1II9UOCS5KG"
# redirect_url = "https://google.com"
# code = "S695H3CSYBXDDZU8JKO6M79QNQES7N6G"
# access_token = "61681706_dc747044a6941fc9aa645a4f3bca2ba5576e7dfb516a3d1889553fe96a4084f6"

# # Fill in your data values

# # Get the authorization access token
# url = "https://api.clickup.com/api/v2/oauth/token"

# query = {
#   "client_id": client_id,
#   "client_secret": client_secret,
#   "code": code,
# }

# response = requests.post(url, params=query)

# data = response.json()
# print(data)

# # Get the authorized user

# import requests

# url = "https://api.clickup.com/api/v2/user"

# headers = {"Authorization": "61681706_dc747044a6941fc9aa645a4f3bca2ba5576e7dfb516a3d1889553fe96a4084f6"}

# response = requests.get(url, headers=headers)

# data = response.json()
# print(data)


# # Get authorized teams

# print("Authorized teams")
# import requests

# url = "https://api.clickup.com/api/v2/team"

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers)

# data = response.json()
# print(data)


# # Get spaces 

# import requests

# team_id = "9013051928"
# url = "https://api.clickup.com/api/v2/team/" + team_id + "/space"

# query = {
#   "archived": "false"
# }

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers, params=query)

# data = response.json()
# print(data)


# # Get folders for workspace
# print("Folders for workspace")
# import requests

# space_id = "90130119692"
# url = "https://api.clickup.com/api/v2/space/" + space_id + "/folder"

# query = {
#   "archived": "false"
# }

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers, params=query)

# data = response.json()
# print(data)


# # Get folderless lists

# print("Folderless lists")

# space_id = "90130119692"
# url = "https://api.clickup.com/api/v2/space/" + space_id + "/list"

# query = {
#   "archived": "false"
# }

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers, params=query)

# data = response.json()
# print(data)

# # Get the list

# print("The List")
# list_id = "901300608424"
# url = "https://api.clickup.com/api/v2/list/" + list_id

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers)

# data = response.json()
# print(data)

# response = requests.get(url, headers=headers)

# # Get task

# import requests
# print("The Task")

# task_id = "86a0t44tq"
# url = "https://api.clickup.com/api/v2/task/" + task_id

# query = {
#   "custom_task_ids": "true",
#   "team_id": "9013051928",
#   "include_subtasks": "true"
# }

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers, params=query)

# data = response.json()
# print(data)


# # Get tasks in a list

# print("Tasks in the list ")
# list_id = "901300608424"
# url = "https://api.clickup.com/api/v2/list/" + list_id + "/task"


# query = {
# #   "statuses": ["open"],
# }

# headers = {"Authorization": access_token}

# response = requests.get(url, headers=headers, params=query)

# data = response.json()
# print(data)


# import requests

# list_id = "901300608424"
# url = "https://api.clickup.com/api/v2/list/" + list_id + "/task"

# query = {
#   "custom_task_ids": "true",
#   "team_id": "9013051928"
# }

# payload = {
#   "name": "New Task Name",
#   "description": "New Task Description",
#   "status": "Open",
#   "priority": 3,
# }

# headers = {
#   "Content-Type": "application/json",
#   "Authorization": "61681706_dc747044a6941fc9aa645a4f3bca2ba5576e7dfb516a3d1889553fe96a4084f6"
# }

# response = requests.post(url, json=payload, headers=headers, params=query)

# data = response.json()
# print(data)



