from typing import List

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_community.tools.todoist.prompt import (
    TODOIST_CLOSE_TASK_PROMPT,
    TODOIST_COMMENT_ADD_PROMPT,
    TODOIST_GET_PROJECTS_PROMPT,
    TODOIST_GET_TASKS_PROMPT,
    TODOIST_MOVE_TASK_PROMPT,
    TODOIST_PROJECT_CREATE_PROMPT,
    TODOIST_TASK_ADD_PROMPT,
)
from langchain_community.tools.todoist.tool import TodoistAction
from langchain_community.utilities.todoist import TodoistAPIWrapper


class TodoistToolkit(BaseToolkit):
    """Todoist Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

    Parameters:
        tools: List[BaseTool]. The tools in the toolkit. Default is an empty list.
    """

    tools: List[BaseTool] = []

    @classmethod
    def from_todoist_api_wrapper(
        cls, todoist_api_wrapper: TodoistAPIWrapper
    ) -> "TodoistToolkit":
        """Create a TodoistToolkit from a TodoistAPIWrapper.

        Args:
            todoist_api_wrapper: TodoistAPIWrapper. The Todoist API wrapper.

        Returns:
            TodoistToolkit. The Todoist toolkit.
        """
        operations = [
            {
                "mode": "get_projects",
                "name": "Get Projects",
                "description": TODOIST_GET_PROJECTS_PROMPT,
            },
            {
                "mode": "get_tasks",
                "name": "Get Tasks",
                "description": TODOIST_GET_TASKS_PROMPT,
            },
            {
                "mode": "create_project",
                "name": "Create Project",
                "description": TODOIST_PROJECT_CREATE_PROMPT,
            },
            {
                "mode": "add_task",
                "name": "Add Task",
                "description": TODOIST_TASK_ADD_PROMPT,
            },
            {
                "mode": "move_task",
                "name": "Move Task",
                "description": TODOIST_MOVE_TASK_PROMPT,
            },
            {
                "mode": "add_comment",
                "name": "Add Comment",
                "description": TODOIST_COMMENT_ADD_PROMPT,
            },
            {
                "mode": "close_task",
                "name": "Close Task",
                "description": TODOIST_CLOSE_TASK_PROMPT,
            },
        ]
        tools = [
            TodoistAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=todoist_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)  # type: ignore[arg-type]

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
