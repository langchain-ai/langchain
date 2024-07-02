"""Integration test for Todoist API Wrapper."""

from datetime import datetime

from langchain_community.tools.todoist.tool import TodoistAction
import pytest

from langchain_community.utilities.todoist import TodoistAPIWrapper

TEST_TODOIST_API_KEY = "573456416febfc01dc3c98fbff38d1b6e5f200a5"

todoist = pytest.importorskip("todoist_api_python")


@pytest.fixture
def todoist_wrapper() -> TodoistAPIWrapper:
    return TodoistAPIWrapper(TEST_TODOIST_API_KEY, ai_task_label="AI")


def test_init(todoist_wrapper: TodoistAPIWrapper) -> None:
    assert isinstance(todoist_wrapper, TodoistAPIWrapper)
    labels = todoist_wrapper.api.get_labels()
    labels[0].name == todoist_wrapper.label


def test_init_with_label() -> None:
    tool = TodoistAPIWrapper(api_key=TEST_TODOIST_API_KEY, ai_task_label="AI Task")
    labels = tool.api.get_labels()
    assert any(label.name == "AI Task" for label in labels)


def test_get_all_projects(todoist_wrapper: TodoistAPIWrapper) -> None:
    projects = todoist_wrapper.get_all_projects()
    assert isinstance(projects, list)


def test_create_project(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    project_name = f"Test Project - {time_str}"
    project_id = None

    try:
        create_response = todoist_wrapper.create_project(project_name)
        assert create_response["name"] == project_name
        project_id = create_response["project_id"]
    finally:
        if project_id:
            todoist_wrapper.api.delete_project(project_id)


def test_add_comment(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"
    task_id = None

    try:
        # Create task
        create_response = todoist_wrapper.api.add_task(content=task_name)
        assert create_response.content == task_name

        # Get task
        task_id = create_response.id
        get_response = todoist_wrapper.api.get_task(task_id)
        assert get_response.content == task_name

        # Add comment
        comment = "This is a comment"
        todoist_wrapper.add_comment(task_id, comment)

        # Verify comment is added
        task = todoist_wrapper.api.get_task(task_id)
        assert task.comment_count == 1

        task_comments = todoist_wrapper.api.get_comments(task_id=task_id)
        assert task_comments[0].content == comment

    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)


def test_create_task(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"
    task_id = None

    try:
        # Create task
        create_response = todoist_wrapper.add_task(content=task_name)
        print(create_response)
        assert create_response.content == task_name
        task_id = create_response.id
        task = todoist_wrapper.api.get_task(task_id)
        assert task.content == task_name
        if todoist_wrapper.label:
            assert task.labels[0] == todoist_wrapper.label
    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)


def test_close_task(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"
    task_id = None

    try:
        # Create task
        create_response = todoist_wrapper.api.add_task(content=task_name)
        task_id = create_response.id

        # Close task
        todoist_wrapper.close_task(task_id)

        # Verify task is closed
        task = todoist_wrapper.api.get_task(task_id)
        assert task.is_completed
    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)


def test_update_task(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"
    task_id = None

    try:
        # Create task
        create_response = todoist_wrapper.add_task(content=task_name)
        assert create_response.content == task_name

        # Update task content
        new_name = f"{task_name} - New"
        todoist_wrapper.update_task(create_response.id, content=new_name, priority=1)

        # Verify task is updated
        task = todoist_wrapper.api.get_task(create_response.id)
        assert task.content == new_name
        assert task.priority == 1
    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)


def test_move_task(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    project_name = f"Test Project - {time_str}"
    task_name = f"Test Task - {time_str}"
    project_id = None
    task_id = None

    try:
        # Create project
        project_response = todoist_wrapper.create_project(project_name)
        assert project_response["name"] == project_name
        project_id = project_response["project_id"]

        # Create task in default project
        create_response = todoist_wrapper.api.add_task(content=task_name)
        task_id = create_response.id

        # Move task to new project
        todoist_wrapper.move_task(task_id, project_id)

        # Verify task is moved
        task = todoist_wrapper.api.get_task(task_id)
        assert task.project_id == project_id
    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)
        if project_id:
            todoist_wrapper.api.delete_project(project_id)


def test_task_related(todoist_wrapper: TodoistAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"
    task_id = None

    try:
        # Create task
        create_response = todoist_wrapper.add_task(content=task_name)
        assert create_response.content == task_name

        # Get task
        task_id = create_response.id
        get_response = todoist_wrapper.api.get_task(task_id)
        assert get_response.content == task_name

        # Update task
        new_name = f"{task_name} - New"
        todoist_wrapper.update_task(task_id, content=new_name)

        get_response_2 = todoist_wrapper.api.get_task(task_id)
        assert get_response_2.content == new_name

        # Add comment
        comment = "This is a comment"
        todoist_wrapper.add_comment(task_id, content=comment)

        task = todoist_wrapper.api.get_task(task_id)
        assert task.comment_count == 1
        task_comments = todoist_wrapper.api.get_comments(task_id=task_id)
        assert task_comments[0].content == comment

        # Close task
        todoist_wrapper.close_task(task_id)
        get_response_3 = todoist_wrapper.api.get_task(task_id)
        assert get_response_3.is_completed

    finally:
        if task_id:
            todoist_wrapper.api.delete_task(task_id)


def test_call_as_action(todoist_wrapper: TodoistAPIWrapper) -> None:
    res = TodoistAction(api_wrapper=todoist_wrapper, mode="get_projects")._run()

    res = TodoistAction(
        api_wrapper=todoist_wrapper,
        mode="add_task",
    )._run(content="test action task")
