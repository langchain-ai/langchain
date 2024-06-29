# flake8: noqa
TODOIST_TASK_ADD_PROMPT = """
    This tool is a wrapper around Todoist's add_task API, useful when you need to create a Todoist task. 
   
   At task creation, the only required field is the content of the task and optionally the projectid. You cannot set any other fields at task creation.
   if you want to set other fields, you can update the task after creation using the update_task tool.
   example input: {"content": "Task content"}
   example input: {"content": "Task content", "project_id": "1234567890"}
    """

TODOIST_PROJECT_CREATE_PROMPT = """
    This tool is a wrapper around Todoist's create_project API, useful when you need to create a Todoist project.
    The input to this tool is a dictionary specifying the fields of a Todoist project, and will be passed to Todoist's `add_project` function.
    Only add fields described by the user.

    Here are a few project descriptions and corresponding input examples:
    Description: create a project with name "General Project"
    Example Input: {{"name": "General Project"}}
    Description: add a new project ("Work Project") with color "blue"
    Example Input: {{"name": "Work Project", "color": "blue"}}
    Description: create a project with name "Project name" and color "red"
    Example Input: {{"name": "Project name", "color": "red"}}
"""

TODOIST_MOVE_TASK_PROMPT = """
    This tool is a wrapper around Todoist's API, useful when you need to move a task to a different project in Todoist.
    Given the task id and the target project id, you want to create a request similar to the following dictionary:
    payload = {{"task_id": "<task_id_to_move>", "project_id": "<target_project_id>"}}

    Here are some example queries and their corresponding payloads:
    Move task 987654321 to project 123456789 -> {{"task_id": "987654321", "project_id": "123456789"}}
    Move task 1122334455 to project 9988776655 -> {{"task_id": "1122334455", "project_id": "9988776655"}}
"""

TODOIST_GET_PROJECTS_PROMPT = """
    This tool is a wrapper around Todoist's API, useful when you need to get all projects that the user is a part of.
    To get a list of all the projects, there are no necessary request parameters. 
"""

TODOIST_GET_TASKS_PROMPT = """
    This tool is a wrapper around Todoist's API, useful when you need to get all tasks that the user is a part of.
    To get a list of all the tasks, there are no necessary request parameters.
"""

TODOIST_UPDATE_TASK_PROMPT = """
    This tool is a wrapper around Todoist's API, 
    useful when you need to update a specific attribute of a task. Given the task id, desired attribute to change, and the new value, you want to create a request similar to the following dictionary:
    payload = {{"task_id": "<task_id_to_update>", "attribute_name": "<attribute_name_to_update>", "value": "<value_to_update_to>"}}

    Here are some example queries and their corresponding payloads:
    Change the content of task 23jn23kjn to "new task content" -> {{"task_id": "23jn23kjn", "attribute_name": "content", "value": "new task content"}}
    Update the priority of task 86a0t44tq to 1 -> {{"task_id": "86a0t44tq", "attribute_name": "priority", "value": 1}}
    Re-write the description of task sdc9ds9jc to 'a new task description' -> {{"task_id": "sdc9ds9jc", "attribute_name": "description", "value": "a new task description"}}
    Change the status of task kjnsdcjc to completed -> {{"task_id": "kjnsdcjc", "attribute_name": "status", "value": "completed"}}
    *IMPORTANT*: Pay attention to the exact syntax above and the correct use of quotes. 
    For changing priority, we expect integers (int).
    For content, description, and status, we expect strings (str).
"""

TODOIST_COMMENT_ADD_PROMPT = """
    This tool is a wrapper around Todoist's API, useful when you need to add a comment to a task.
    Given the task id and the content of the comment, you want to create a request similar to the following dictionary:
    payload = {{"task_id": "<task_id_to_comment_on>", "content": "<comment_content>"}}
    """

TODOIST_CLOSE_TASK_PROMPT = """
    This tool is a wrapper around Todoist's API, useful when you need to close a task.
    Given the task id, you want to create a request similar to the following dictionary:
    payload = {{"task_id": "<task_id_to_close>"}}
    """
