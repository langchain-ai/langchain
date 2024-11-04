import json
from datetime import date, datetime
from decimal import Decimal
from hashlib import md5
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import BaseToolkit
from pydantic import BaseModel, Field, create_model
from typing_extensions import Self

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import FunctionInfo

from pydantic import ConfigDict

from langchain_community.tools.databricks._execution import execute_function


def _uc_type_to_pydantic_type(uc_type_json: Union[str, Dict[str, Any]]) -> Type:
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
        tpe = uc_type_json["type"]
        if tpe == "array":
            element_type = _uc_type_to_pydantic_type(uc_type_json["elementType"])
            if uc_type_json["containsNull"]:
                element_type = Optional[element_type]  # type: ignore
            return List[element_type]  # type: ignore
        elif tpe == "map":
            key_type = uc_type_json["keyType"]
            assert key_type == "string", TypeError(
                f"Only support STRING key type for MAP but got {key_type}."
            )
            value_type = _uc_type_to_pydantic_type(uc_type_json["valueType"])
            if uc_type_json["valueContainsNull"]:
                value_type: Type = Optional[value_type]  # type: ignore
            return Dict[str, value_type]  # type: ignore
        elif tpe == "struct":
            fields = {}
            for field in uc_type_json["fields"]:
                field_type = _uc_type_to_pydantic_type(field["type"])
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


def _generate_args_schema(function: "FunctionInfo") -> Type[BaseModel]:
    if function.input_params is None:
        return BaseModel
    params = function.input_params.parameters
    assert params is not None
    fields = {}
    for p in params:
        assert p.type_json is not None
        type_json = json.loads(p.type_json)["type"]
        pydantic_type = _uc_type_to_pydantic_type(type_json)
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


def _get_tool_name(function: "FunctionInfo") -> str:
    tool_name = f"{function.catalog_name}__{function.schema_name}__{function.name}"[
        -64:
    ]
    return tool_name


def _get_default_workspace_client() -> Any:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as e:
        raise ImportError(
            "Could not import databricks-sdk python package. "
            "Please install it with `pip install databricks-sdk`."
        ) from e
    return WorkspaceClient()


class UCFunctionToolkit(BaseToolkit):
    warehouse_id: str = Field(
        description="The ID of a Databricks SQL Warehouse to execute functions."
    )

    workspace_client: Any = Field(
        default_factory=_get_default_workspace_client,
        description="Databricks workspace client.",
    )

    tools: Dict[str, BaseTool] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def include(self, *function_names: str, **kwargs: Any) -> Self:
        """
        Includes UC functions to the toolkit.

        Args:
            functions: A list of UC function names in the format
                "catalog_name.schema_name.function_name" or
                "catalog_name.schema_name.*".
                If the function name ends with ".*",
                all functions in the schema will be added.
            kwargs: Extra arguments to pass to StructuredTool, e.g., `return_direct`.
        """
        for name in function_names:
            if name.endswith(".*"):
                catalog_name, schema_name = name[:-2].split(".")
                # TODO: handle pagination, warn and truncate if too many
                functions = self.workspace_client.functions.list(
                    catalog_name=catalog_name, schema_name=schema_name
                )
                for f in functions:
                    assert f.full_name is not None
                    self.include(f.full_name, **kwargs)
            else:
                if name not in self.tools:
                    self.tools[name] = self._make_tool(name, **kwargs)
        return self

    def _make_tool(self, function_name: str, **kwargs: Any) -> BaseTool:
        function = self.workspace_client.functions.get(function_name)
        name = _get_tool_name(function)
        description = function.comment or ""
        args_schema = _generate_args_schema(function)

        def func(*args: Any, **kwargs: Any) -> str:
            # TODO: We expect all named args and ignore args.
            # Non-empty args show up when the function has no parameters.
            args_json = json.loads(json.dumps(kwargs, default=str))
            result = execute_function(
                ws=self.workspace_client,
                warehouse_id=self.warehouse_id,
                function=function,
                parameters=args_json,
            )
            return result.to_json()

        return StructuredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            func=func,
            **kwargs,
        )

    def get_tools(self) -> List[BaseTool]:
        return list(self.tools.values())
