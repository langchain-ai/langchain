from unittest import mock

import pytest

from langchain_community.tools.databricks._execution import (
    DEFAULT_EXECUTE_FUNCTION_ARGS,
    EXECUTE_FUNCTION_ARG_NAME,
    execute_function,
)


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
def test_execute_function(parameters, execute_params) -> None:
    workspace_client = mock.Mock()

    def mock_execute_statement(*args, **kwargs):
        for key, value in execute_params.items():
            assert kwargs[key] == value
        return mock.Mock()

    workspace_client.statement_execution.execute_statement = mock_execute_statement

    function = mock.Mock()
    function.data_type = "TABLE_TYPE"
    function.input_params.parameters = []
    execute_function(
        workspace_client, warehouse_id="id", function=function, parameters=parameters
    )
