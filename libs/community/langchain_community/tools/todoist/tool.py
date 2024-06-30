from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import Field

from langchain_community.utilities.todoist import TodoistAPIWrapper


class TodoistAction(BaseTool):
    """A tool for interacting with Todoist."""

    mode: str
    name: str = "todoist"
    description: str = "A wrapper around Todoist API."
    api_wrapper: TodoistAPIWrapper = Field(default_factory=TodoistAPIWrapper)

    def _run(self, *args, **kwargs) -> Any:
        if self.mode == "get_projects":
            return self.api_wrapper.get_all_projects()
        elif self.mode == "get_tasks":
            return self.api_wrapper.get_all_tasks()
        elif self.mode == "create_project":
            name = kwargs.get("name")
            if not name:
                raise ValueError("Project name must be provided")
            return self.api_wrapper.create_project(name)
        elif self.mode == "add_task":
            content = kwargs.get("content")
            project_id = kwargs.get("project_id")
            if not content:
                raise ValueError("Content must be provided")
            return self.api_wrapper.add_task(content, project_id)
        elif self.mode == "close_task":
            task_id = kwargs.get("task_id")
            if not task_id:
                raise ValueError("Task ID must be provided")
        elif self.mode == "add_comment":
            task_id = kwargs.get("task_id")
            content = kwargs.get("content")
            if not task_id or not content:
                raise ValueError("Task ID and Content must be provided")
            return self.api_wrapper.add_comment(task_id, content)

        elif self.mode == "move_task":
            task_id = kwargs.get("task_id")
            project_id = kwargs.get("project_id")
            if not task_id or not project_id:
                raise ValueError("Task ID and Project ID must be provided")
            return self.api_wrapper.move_task(task_id, project_id)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    async def _arun(self, *args, **kwargs) -> Any:
        return self._run(*args, **kwargs)
