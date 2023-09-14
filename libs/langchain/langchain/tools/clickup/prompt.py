# flake8: noqa
CLICKUP_TASK_CREATE_PROMPT = """
    This tool is a wrapper around clickup's create_task API, useful when you need to create a CLICKUP task. 
    The input to this tool is a dictionary specifying the fields of the CLICKUP task, and will be passed into clickup's CLICKUP `create_task` function.
    For example, to create a task with priority 3 called "New Task Name" with description "New Task Description", with status "open" you would pass in the following dictionary: 
    payload = {{
        "name": "New Task Name",
        "description": "New Task Description",
        "status": "Open",
        "priority": 3,
    }}
    """

CLICKUP_GET_TASK_PROMPT = """
    This tool is a wrapper around clickup's API, 
    useful when you need to get a specific task for the user. Given the task id you want to create a request similar to the following dictionary:
    payload = {{"task_id": "86a0t44tq"}}
    """

CLICKUP_GET_ALL_TEAMS_PROMPT = """
    This tool is a wrapper around clickup's API, useful when you need to get all teams that the user is a part of.
    To get a list of all the teams there is no necessary request parameters. 
    """

CLICKUP_GET_LIST_PROMPT = """
    This tool is a wrapper around clickup's API, 
    useful when you need to get a specific list for the user. Given the list id you want to create a request similar to the following dictionary:
    payload = {{"list_id": "901300608424"}}
    """

CLICKUP_GET_FOLDERS_PROMPT = """
    This tool is a wrapper around clickup's API, 
    useful when you need to get a specific folder for the user. Given the user's workspace id you want to create a request similar to the following dictionary:
    payload = {{"space_id": "90130119692"}}
    """
    
CLICKUP_UPDATE_TASK_PROMPT = """
    This tool is a wrapper around clickup's API, 
    useful when you need to update a specific attribute of a task. Given the task id, desired attribute to change and the new value you want to create a request similar to the following dictionary:
    payload = {{"task_id": "86a0t44tq", "attribute_name": "priority", "new_value": "1"}}
    """


