import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import FunctionInfo
    from databricks.sdk.service.sql import StatementParameterListItem, StatementState

EXECUTE_FUNCTION_ARG_NAME = "__execution_args__"
DEFAULT_EXECUTE_FUNCTION_ARGS = {
    "wait_timeout": "30s",
    "row_limit": 100,
    "byte_limit": 4096,
}
UC_TOOL_CLIENT_EXECUTION_TIMEOUT = "UC_TOOL_CLIENT_EXECUTION_TIMEOUT"
DEFAULT_UC_TOOL_CLIENT_EXECUTION_TIMEOUT = "120"
_logger = logging.getLogger(__name__)


def is_scalar(function: "FunctionInfo") -> bool:
    from databricks.sdk.service.catalog import ColumnTypeName

    return function.data_type != ColumnTypeName.TABLE_TYPE


@dataclass
class ParameterizedStatement:
    statement: str
    parameters: List["StatementParameterListItem"]


@dataclass
class FunctionExecutionResult:
    """
    Result of executing a function.
    We always use a string to present the result value for AI model to consume.
    """

    error: Optional[str] = None
    format: Optional[Literal["SCALAR", "CSV"]] = None
    value: Optional[str] = None
    truncated: Optional[bool] = None

    def to_json(self) -> str:
        data = {k: v for (k, v) in self.__dict__.items() if v is not None}
        return json.dumps(data)


def get_execute_function_sql_stmt(
    function: "FunctionInfo", json_params: Dict[str, Any]
) -> ParameterizedStatement:
    from databricks.sdk.service.catalog import ColumnTypeName
    from databricks.sdk.service.sql import StatementParameterListItem

    parts = []
    output_params = []
    if is_scalar(function):
        # TODO: IDENTIFIER(:function) did not work
        parts.append(f"SELECT {function.full_name}(")
    else:
        parts.append(f"SELECT * FROM {function.full_name}(")
    if function.input_params is None or function.input_params.parameters is None:
        assert (
            not json_params
        ), "Function has no parameters but parameters were provided."
    else:
        args = []
        use_named_args = False
        for p in function.input_params.parameters:
            if p.name not in json_params:
                if p.parameter_default is not None:
                    use_named_args = True
                else:
                    raise ValueError(
                        f"Parameter {p.name} is required but not provided."
                    )
            else:
                arg_clause = ""
                if use_named_args:
                    arg_clause += f"{p.name} => "
                json_value = json_params[p.name]
                if p.type_name in (
                    ColumnTypeName.ARRAY,
                    ColumnTypeName.MAP,
                    ColumnTypeName.STRUCT,
                ):
                    # Use from_json to restore values of complex types.
                    json_value_str = json.dumps(json_value)
                    # TODO: parametrize type
                    arg_clause += f"from_json(:{p.name}, '{p.type_text}')"
                    output_params.append(
                        StatementParameterListItem(name=p.name, value=json_value_str)
                    )
                elif p.type_name == ColumnTypeName.BINARY:
                    # Use ubbase64 to restore binary values.
                    arg_clause += f"unbase64(:{p.name})"
                    output_params.append(
                        StatementParameterListItem(name=p.name, value=json_value)
                    )
                else:
                    arg_clause += f":{p.name}"
                    output_params.append(
                        StatementParameterListItem(
                            name=p.name, value=json_value, type=p.type_text
                        )
                    )
                args.append(arg_clause)
        parts.append(",".join(args))
    parts.append(")")
    # TODO: check extra params in kwargs
    statement = "".join(parts)
    return ParameterizedStatement(statement=statement, parameters=output_params)


def execute_function(
    ws: "WorkspaceClient",
    warehouse_id: str,
    function: "FunctionInfo",
    parameters: Dict[str, Any],
) -> FunctionExecutionResult:
    """
    Execute a function with the given arguments and return the result.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Could not import pandas python package. "
            "Please install it with `pip install pandas`."
        ) from e
    from databricks.sdk.service.sql import StatementState

    if (
        function.input_params
        and function.input_params.parameters
        and any(
            p.name == EXECUTE_FUNCTION_ARG_NAME
            for p in function.input_params.parameters
        )
    ):
        raise ValueError(
            "Parameter name conflicts with the reserved argument name for executing "
            f"functions: {EXECUTE_FUNCTION_ARG_NAME}. "
            f"Please rename the parameter {EXECUTE_FUNCTION_ARG_NAME}."
        )

    # avoid modifying the original dict
    execute_statement_args = {**DEFAULT_EXECUTE_FUNCTION_ARGS}
    allowed_execute_statement_args = inspect.signature(
        ws.statement_execution.execute_statement
    ).parameters
    if not any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in allowed_execute_statement_args.values()
    ):
        invalid_params = set()
        passed_execute_statement_args = parameters.pop(EXECUTE_FUNCTION_ARG_NAME, {})
        for k, v in passed_execute_statement_args.items():
            if k in allowed_execute_statement_args:
                execute_statement_args[k] = v
            else:
                invalid_params.add(k)
        if invalid_params:
            raise ValueError(
                f"Invalid parameters for executing functions: {invalid_params}. "
                f"Allowed parameters are: {allowed_execute_statement_args.keys()}."
            )

    # TODO: async so we can run functions in parallel
    parametrized_statement = get_execute_function_sql_stmt(function, parameters)
    response = ws.statement_execution.execute_statement(
        statement=parametrized_statement.statement,
        warehouse_id=warehouse_id,
        parameters=parametrized_statement.parameters,
        **execute_statement_args,  # type: ignore
    )
    if response.status and job_pending(response.status.state) and response.statement_id:
        statement_id = response.statement_id
        wait_time = 0
        retry_cnt = 0
        client_execution_timeout = int(
            os.environ.get(
                UC_TOOL_CLIENT_EXECUTION_TIMEOUT,
                DEFAULT_UC_TOOL_CLIENT_EXECUTION_TIMEOUT,
            )
        )
        while wait_time < client_execution_timeout:
            wait = min(2**retry_cnt, client_execution_timeout - wait_time)
            _logger.debug(
                f"Retrying {retry_cnt} time to get statement execution "
                f"status after {wait} seconds."
            )
            time.sleep(wait)
            response = ws.statement_execution.get_statement(statement_id)  # type: ignore
            if response.status is None or not job_pending(response.status.state):
                break
            wait_time += wait
            retry_cnt += 1
        if response.status and job_pending(response.status.state):
            return FunctionExecutionResult(
                error=f"Statement execution is still pending after {wait_time} "
                "seconds. Please increase the wait_timeout argument for executing "
                f"the function or increase {UC_TOOL_CLIENT_EXECUTION_TIMEOUT} "
                "environment variable for increasing retrying time, default is "
                f"{DEFAULT_UC_TOOL_CLIENT_EXECUTION_TIMEOUT} seconds."
            )
    assert response.status is not None, f"Statement execution failed: {response}"
    if response.status.state != StatementState.SUCCEEDED:
        error = response.status.error
        assert (
            error is not None
        ), f"Statement execution failed but no error message was provided: {response}"
        return FunctionExecutionResult(error=f"{error.error_code}: {error.message}")
    manifest = response.manifest
    assert manifest is not None
    truncated = manifest.truncated
    result = response.result
    assert (
        result is not None
    ), "Statement execution succeeded but no result was provided."
    data_array = result.data_array
    if is_scalar(function):
        value = None
        if data_array and len(data_array) > 0 and len(data_array[0]) > 0:
            value = str(data_array[0][0])  # type: ignore
        return FunctionExecutionResult(
            format="SCALAR", value=value, truncated=truncated
        )
    else:
        schema = manifest.schema
        assert (
            schema is not None and schema.columns is not None
        ), "Statement execution succeeded but no schema was provided."
        columns = [c.name for c in schema.columns]
        if data_array is None:
            data_array = []
        pdf = pd.DataFrame.from_records(data_array, columns=columns)
        csv_buffer = StringIO()
        pdf.to_csv(csv_buffer, index=False)
        return FunctionExecutionResult(
            format="CSV", value=csv_buffer.getvalue(), truncated=truncated
        )


def job_pending(state: Optional["StatementState"]) -> bool:
    from databricks.sdk.service.sql import StatementState

    return state in (StatementState.PENDING, StatementState.RUNNING)
