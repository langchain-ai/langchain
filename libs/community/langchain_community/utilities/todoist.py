import os
import uuid
from datetime import datetime
from functools import cache
from typing import Any, Dict, List, Optional

import pydantic as pydantic
import requests
from dateutil import parser
from todoist_api_python.api import Project, TodoistAPI
from todoist_api_python.models import Task


def create_human_friendly_date(date: str) -> str:
    input_datetime = parser.isoparse(date)
    now = datetime.now().replace(tzinfo=input_datetime.tzinfo)
    diff = now - input_datetime

    if diff.total_seconds() > 24 * 3600:
        return f"{int(diff.total_seconds() / (24 * 3600))} days ago"
    elif diff.total_seconds() > 3600:
        return f"{int(diff.total_seconds() / 3600)} hours ago"
    elif diff.total_seconds() > 60:
        return f"{int(diff.total_seconds() / 60)} minutes ago"

    return "Just now"


class TodoistAPIWrapper:
    def __init__(
        self, api_key: Optional[str] = None, ai_task_label: Optional[str] = None
    ) -> None:
        self.api_key = api_key or os.getenv("TODOIST_API_KEY")
        self.label = ai_task_label
        if not self.api_key:
            raise ValueError("Todoist API Key must be provided or set in environment")
        self.api = TodoistAPI(self.api_key)

        if ai_task_label is not None:
            labels = self.api.get_labels()
            if not any(label.name == ai_task_label for label in labels):
                self.api.add_label(name=ai_task_label, color="lime_green")

    @property
    @cache
    def inbox_id(self) -> str:
        projects = self.api.get_projects()
        for project in projects:
            if project.name.lower() == "inbox":
                return project.id
        raise ValueError("No inbox found")

    @property
    def _todoist_project_id_to_project_name(self) -> Dict[str, str]:
        todoist_projects = self.api.get_projects()
        return {project.id: project.name for project in todoist_projects}

    def get_all_projects(self) -> List[Dict[str, Any]]:
        return [self._format_project(project) for project in self.api.get_projects()]

    def _format_project(self, project: Project) -> Dict[str, str]:
        return {
            "name": project.name,
            "project_id": project.id,
            "is_inbox": project.is_inbox_project,
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return [
            task
            for task in self._get_all_tasks()
            if task["project_id"] != self.inbox_id
        ]

    def _get_all_tasks(self) -> List[Dict[str, str]]:
        results = []
        for task in self.api.get_tasks():
            results.append(
                {
                    "name": task.content,
                    "task_id": task.id,
                    "project_id": task.project_id,
                    "created": create_human_friendly_date(task.created_at),
                    "project_name": self._todoist_project_id_to_project_name[
                        task.project_id
                    ],
                }
            )
        return results

    def create_project(self, name: str) -> Dict[str, Any]:
        for project in self.api.get_projects():
            if project.name.lower() == name.lower():
                raise ValueError(f"Project {name} already exists.")
        project = self.api.add_project(name)
        return self._format_project(project)

    def add_task(self, content: str, project_id: Optional[str] = None) -> Task:
        task = self.api.add_task(
            content=content,
            project_id=project_id,
        )
        if self.label:
            self.update_task(task.id, labels=[self.label])
        return task

    def update_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        return self.api.update_task(task_id, **kwargs)

    def add_comment(self, task_id: str, content: str) -> Dict[str, Any]:
        return self.api.add_comment(content=content, task_id=task_id)

    def close_task(self, task_id: str) -> None:
        return self.api.close_task(task_id)

    def move_task(self, task_id: str, project_id: str) -> None:
        task = self._get_task(task_id)
        _project = self._get_project(project_id)

        if task["project_id"] == project_id:
            raise ValueError(
                f"Task {task_id} is already in project {project_id}. No need to move it."
            )

        return self._move_task_api_call(task_id, project_id)

    def _move_task_api_call(self, task_id: str, project_id: str):
        body = {
            "commands": [
                {
                    "type": "item_move",
                    "args": {"id": task_id, "project_id": project_id},
                    "uuid": uuid.uuid4().hex,
                },
            ],
        }
        response = requests.post(
            "https://api.todoist.com/sync/v9/sync",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        if response.status_code >= 400:
            raise ValueError(
                f"Error failed to move task {task_id} to project {project_id}. Error: {response.text}"
            )

        return response.json()

    def _get_task(self, task_id) -> Dict[str, str]:
        for task in self._get_all_tasks():
            if task["task_id"] == task_id:
                return task
        raise ValueError(f"Task {task_id} does not exist.")

    def _get_project(self, project_id) -> Dict[str, str]:
        for project in self.get_all_projects():
            if project["project_id"] == project_id:
                return project
        raise ValueError(f"Project {project_id} does not exist.")
