from typing import Any, Optional
from unittest import mock

import pytest

from langchain_community.tools.databricks._execution import (
    DEFAULT_EXECUTE_FUNCTION_ARGS,
    EXECUTE_FUNCTION_ARG_NAME,
    execute_function,
)


@pytest.mark.requires("databricks.sdk")
@pytest.mark.parametrize(
    ("parameters", "execute_params"),
    [
        ({"a": 1, "b": 2}, DEFAULT_EXECUTE_FUNCTION_ARGS),
        (
            {"a": 1, EXECUTE_FUNCTION_ARG_NAME: {"wait_timeout": "10s"}},
            {**DEFAULT_EXECUTE_FUNCTION_ARGS, "wait_timeout": "10s"},
        ),
        (
            {EXECUTE_FUNCTION_ARG_NAME: {"row_limit": "1000"}},
            {**DEFAULT_EXECUTE_FUNCTION_ARGS, "row_limit": "1000"},
        ),
    ],
)
def test_execute_function(parameters: dict, execute_params: dict) -> None:
    workspace_client = mock.Mock()

    def mock_execute_statement(
        statement: str,
        warehouse_id: str,
        *,
        byte_limit: Optional[int] = None,
        catalog: Optional[str] = None,
        disposition: Optional[Any] = None,
        format: Optional[Any] = None,
        on_wait_timeout: Optional[Any] = None,
        parameters: Optional[list[Any]] = None,
        row_limit: Optional[int] = None,
        schema: Optional[str] = None,
        wait_timeout: Optional[str] = None,
    ) -> mock.Mock:
        for key, value in execute_params.items():
            assert locals()[key] == value
        return mock.Mock()

    workspace_client.statement_execution.execute_statement = mock_execute_statement
    function = mock.Mock()
    function.data_type = "TABLE_TYPE"
    function.input_params.parameters = []
    execute_function(
        workspace_client, warehouse_id="id", function=function, parameters=parameters
    )


@pytest.mark.requires("databricks.sdk")
def test_execute_function_error() -> None:
    workspace_client = mock.Mock()

    def mock_execute_statement(
        statement: str,
        warehouse_id: str,
        *,
        byte_limit: Optional[int] = None,
        catalog: Optional[str] = None,
        disposition: Optional[Any] = None,
        format: Optional[Any] = None,
        on_wait_timeout: Optional[Any] = None,
        parameters: Optional[list[Any]] = None,
        row_limit: Optional[int] = None,
        schema: Optional[str] = None,
        wait_timeout: Optional[str] = None,
    ) -> mock.Mock:
        return mock.Mock()

    workspace_client.statement_execution.execute_statement = mock_execute_statement
    function = mock.Mock()
    function.data_type = "TABLE_TYPE"
    function.input_params.parameters = []
    parameters = {EXECUTE_FUNCTION_ARG_NAME: {"invalid_param": "123"}}
    with pytest.raises(
        ValueError,
        match=r"Invalid parameters for executing functions: {'invalid_param'}. ",
    ):
        execute_function(
            workspace_client,
            warehouse_id="id",
            function=function,
            parameters=parameters,
        )
