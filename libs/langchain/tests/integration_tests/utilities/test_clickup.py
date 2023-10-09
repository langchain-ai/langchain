"""Integration test for JIRA API Wrapper."""
import json
from datetime import datetime

import pytest

from langchain.utilities.clickup import ClickupAPIWrapper


@pytest.fixture
def clickup_wrapper() -> ClickupAPIWrapper:
    return ClickupAPIWrapper()


def test_init(clickup_wrapper: ClickupAPIWrapper) -> None:
    assert isinstance(clickup_wrapper, ClickupAPIWrapper)


def test_get_access_code_url() -> None:
    assert isinstance(
        ClickupAPIWrapper.get_access_code_url("oauth_client_id", "oauth_client_secret"),
        str,
    )


def test_get_access_token() -> None:
    output = ClickupAPIWrapper.get_access_token(
        "oauth_client_id", "oauth_client_secret", "code"
    )
    assert output is None


def test_folder_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Folder - {time_str}"

    # Create Folder
    create_response = json.loads(
        clickup_wrapper.run(mode="create_folder", query=json.dumps({"name": task_name}))
    )
    assert create_response["name"] == task_name


def test_list_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test List - {time_str}"

    # Create List
    create_response = json.loads(
        clickup_wrapper.run(mode="create_list", query=json.dumps({"name": task_name}))
    )
    assert create_response["name"] == task_name


def test_task_related(clickup_wrapper: ClickupAPIWrapper) -> None:
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    task_name = f"Test Task - {time_str}"

    # Create task
    create_response = json.loads(
        clickup_wrapper.run(
            mode="create_task",
            query=json.dumps({"name": task_name, "description": "This is a Test"}),
        )
    )
    assert create_response["name"] == task_name

    # Get task
    task_id = create_response["id"]
    get_response = json.loads(
        clickup_wrapper.run(mode="get_task", query=json.dumps({"task_id": task_id}))
    )

    assert get_response["name"] == task_name

    # Update task
    new_name = f"{task_name} - New"
    clickup_wrapper.run(
        mode="update_task",
        query=json.dumps(
            {"task_id": task_id, "attribute_name": "name", "value": new_name}
        ),
    )

    get_response_2 = json.loads(
        clickup_wrapper.run(mode="get_task", query=json.dumps({"task_id": task_id}))
    )
    assert get_response_2["name"] == new_name
