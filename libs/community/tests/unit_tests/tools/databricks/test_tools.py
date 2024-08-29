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

    def mock_execute_statement(  # type: ignore
        statement,
        warehouse_id,
        *,
        byte_limit=None,
        catalog=None,
        disposition=None,
        format=None,
        on_wait_timeout=None,
        parameters=None,
        row_limit=None,
        schema=None,
        wait_timeout=None,
    ):
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

    def mock_execute_statement(  # type: ignore
        statement,
        warehouse_id,
        *,
        byte_limit=None,
        catalog=None,
        disposition=None,
        format=None,
        on_wait_timeout=None,
        parameters=None,
        row_limit=None,
        schema=None,
        wait_timeout=None,
    ):
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
