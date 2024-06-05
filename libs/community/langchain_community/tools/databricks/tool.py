import json
from datetime import date, datetime
from decimal import Decimal
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import FunctionInfo

from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.tools import BaseTool

from langchain_community.tools.databricks.execution import (
    FunctionExecutionResult,
    execute_function,
)


def uc_type_to_pydantic_type(uc_type_json: Union[str, Dict[str, Any]]) -> Type:
    mapping = {
        "long": int,
        "binary": bytes,
        "boolean": bool,
        "date": date,
        "double": float,
        "float": float,
        "integer": int,
        "short": int,
        "string": str,
        "timestamp": datetime,
        "timestamp_ntz": datetime,
        "byte": int,
    }
    if isinstance(uc_type_json, str):
        if uc_type_json in mapping:
            return mapping[uc_type_json]
        else:
            if uc_type_json.startswith("decimal"):
                return Decimal
            elif uc_type_json == "void" or uc_type_json.startswith("interval"):
                raise TypeError(f"Type {uc_type_json} is not supported.")
            else:
                raise TypeError(
                    f"Unknown type {uc_type_json}. Try upgrading this package."
                )
    else:
        assert isinstance(uc_type_json, dict)
        type = uc_type_json["type"]
        if type == "array":
            element_type = uc_type_to_pydantic_type(uc_type_json["elementType"])
            if uc_type_json["containsNull"]:
                element_type = Optional[element_type]  # type: ignore
            return List[element_type]  # type: ignore
        elif type == "map":
            key_type = uc_type_json["keyType"]
            assert key_type == "string", TypeError(
                f"Only support STRING key type for MAP but got {key_type}."
            )
            value_type = uc_type_to_pydantic_type(uc_type_json["valueType"])
            if uc_type_json["valueContainsNull"]:
                value_type: Type = Optional[value_type]  # type: ignore
            return Dict[str, value_type]  # type: ignore
        elif type == "struct":
            fields = {}
            for field in uc_type_json["fields"]:
                field_type = uc_type_to_pydantic_type(field["type"])
                if field.get("nullable"):
                    field_type = Optional[field_type]  # type: ignore
                comment = (
                    uc_type_json["metadata"].get("comment")
                    if "metadata" in uc_type_json
                    else None
                )
                fields[field["name"]] = (field_type, Field(..., description=comment))
            uc_type_json_str = json.dumps(uc_type_json, sort_keys=True)
            type_hash = md5(uc_type_json_str.encode()).hexdigest()[:8]
            return create_model(f"Struct_{type_hash}", **fields)  # type: ignore
        else:
            raise TypeError(f"Unknown type {uc_type_json}. Try upgrading this package.")


def generate_args_schema(function: FunctionInfo) -> Optional[Type[BaseModel]]:
    if function.input_params is None:
        return None
    params = function.input_params.parameters
    assert params is not None
    fields = {}
    for p in params:
        assert p.type_json is not None
        type_json = json.loads(p.type_json)["type"]
        pydantic_type = uc_type_to_pydantic_type(type_json)
        description = p.comment
        default: Any = ...
        if p.parameter_default:
            pydantic_type = Optional[pydantic_type]  # type: ignore
            default = None
            # TODO: Convert default value string to the correct type.
            # We might need to use statement execution API
            # to get the JSON representation of the value.
            default_description = f"(Default: {p.parameter_default})"
            if description:
                description += f" {default_description}"
            else:
                description = default_description
        fields[p.name] = (
            pydantic_type,
            Field(default=default, description=description),
        )
    return create_model(
        f"{function.catalog_name}__{function.schema_name}__{function.name}__params",
        **fields,  # type: ignore
    )


def get_tool_name(function: FunctionInfo) -> str:
    tool_name = f"{function.catalog_name}__{function.schema_name}__{function.name}"[
        -64:
    ]
    return tool_name


class UCFunctionTool(BaseTool):
    """Tool for calling a user-defined function in Unity Catalog."""

    function: FunctionInfo
    warehouse_id: str
    workspace_client: WorkspaceClient

    def __init__(
        self,
        function_name: str,
        warehouse_id: str,
        workspace_client: Optional[WorkspaceClient] = None,
        **kwargs: Any,
    ):
        if workspace_client is None:
            workspace_client = WorkspaceClient()
        function = workspace_client.functions.get(function_name)
        name = get_tool_name(function)
        description = function.comment or ""
        args_schema = generate_args_schema(function)
        super().__init__(  # type: ignore
            function=function,
            warehouse_id=warehouse_id,
            workspace_client=workspace_client,
            name=name,
            description=description,
            args_schema=args_schema,
            **kwargs,
        )

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> FunctionExecutionResult:
        """Call the user-defined function."""
        # TODO: We expect all named args and ignore args.
        # Non-empty args show up when the function has no parameters.
        args_json = json.loads(json.dumps(kwargs, default=str))
        return execute_function(
            ws=self.workspace_client,
            warehouse_id=self.warehouse_id,
            function=self.function,
            parameters=args_json,
        )


class UCFunctionTools:
    def __init__(
        self, *, warehouse_id: str, workspace_client: Optional[WorkspaceClient] = None
    ):
        if workspace_client is None:
            workspace_client = WorkspaceClient()
        # Check if the warehouse exists
        workspace_client.warehouses.get(warehouse_id)
        self.workspace_client = workspace_client
        self.warehouse_id = warehouse_id

    def get_tool(self, function_name: str) -> UCFunctionTool:
        """
        Gets a UC function as tool.

        Args:
            function_name: The UC function name in the format
                "catalog_name.schema_name.function_name".
        """
        return UCFunctionTool(
            function_name=function_name,
            warehouse_id=self.warehouse_id,
            workspace_client=self.workspace_client,
        )

    def get_tools(self, function_names: List[str]) -> List[UCFunctionTool]:
        """
        Gets UC functions as tools.

        Args:
            function_names: A list of UC function names in the format
                "catalog_name.schema_name.function_name" or
                "catalog_name.schema_name.*".
                If the function name ends with ".*",
                all functions in the schema will be added.
        """
        assert isinstance(
            function_names, list
        ), f"Function names must be a list but got {type(function_names)}."
        added = set()
        tools = []
        for name in function_names:
            assert isinstance(
                name, str
            ), f"Function name must be a string but got {type(name)}."
            if name.endswith(".*"):
                catalog_name, schema_name = name[:-2].split(".")
                # TODO: handle pagination, warn and truncate if too many
                functions = self.workspace_client.functions.list(
                    catalog_name=catalog_name, schema_name=schema_name
                )
                for f in functions:
                    assert f.full_name is not None
                    if f.full_name not in added:
                        added.add(f.full_name)
                        tools.append(self.get_tool(f.full_name))
            else:
                if name not in added:
                    added.add(name)
                    tools.append(self.get_tool(name))
        return tools
