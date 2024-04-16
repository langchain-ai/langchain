import json
from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

ANYSDK_CRUD_CONTROLS_CREATE = "False"
ANYSDK_CRUD_CONTROLS_READ = "True"
ANYSDK_CRUD_CONTROLS_UPDATE = "False"
ANYSDK_CRUD_CONTROLS_DELETE = "False"

ANYSDK_CRUD_CONTROLS_CREATE_LIST = "create"
ANYSDK_CRUD_CONTROLS_READ_LIST = "get,read,list"
ANYSDK_CRUD_CONTROLS_UPDATE_LIST = "update,put,post"
ANYSDK_CRUD_CONTROLS_DELETE_LIST = "delete,destroy,remove"


class AnySdkWrapper(BaseModel):
    client: Any
    operations: List[Dict] = []
    crud_controls_create: Optional[str] = ANYSDK_CRUD_CONTROLS_CREATE
    crud_controls_create_list: Optional[str] = ANYSDK_CRUD_CONTROLS_CREATE_LIST
    crud_controls_read: Optional[str] = ANYSDK_CRUD_CONTROLS_READ
    crud_controls_read_list: Optional[str] = ANYSDK_CRUD_CONTROLS_READ_LIST
    crud_controls_update: Optional[str] = ANYSDK_CRUD_CONTROLS_UPDATE
    crud_controls_update_list: Optional[str] = ANYSDK_CRUD_CONTROLS_UPDATE_LIST
    crud_controls_delete: Optional[str] = ANYSDK_CRUD_CONTROLS_DELETE
    crud_controls_delete_list: Optional[str] = ANYSDK_CRUD_CONTROLS_DELETE_LIST

    class Config:
        extra = Extra.allow

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        crud_controls_create = get_from_dict_or_env(
            values,
            "crud_controls_create",
            "ANYSDK_CRUD_CONTROLS_CREATE",
            default=ANYSDK_CRUD_CONTROLS_CREATE,
        )
        values["crud_controls_create"] = bool(crud_controls_create)

        crud_controls_create_list = get_from_dict_or_env(
            values,
            "crud_controls_create_list",
            "ANYSDK_CRUD_CONTROLS_CREATE_LIST",
            default=ANYSDK_CRUD_CONTROLS_CREATE_LIST,
        )
        values["crud_controls_create_list"] = crud_controls_create_list.split(",")

        crud_controls_read = get_from_dict_or_env(
            values,
            "crud_controls_read",
            "ANYSDK_CRUD_CONTROLS_READ",
            default=ANYSDK_CRUD_CONTROLS_READ,
        )
        values["crud_controls_read"] = bool(crud_controls_read)

        crud_controls_read_list = get_from_dict_or_env(
            values,
            "crud_controls_read_list",
            "ANYSDK_CRUD_CONTROLS_READ_LIST",
            default=ANYSDK_CRUD_CONTROLS_READ_LIST,
        )
        values["crud_controls_read_list"] = crud_controls_read_list.split(",")

        crud_controls_update = get_from_dict_or_env(
            values,
            "crud_controls_update",
            "ANYSDK_CRUD_CONTROLS_UPDATE",
            default=bool(ANYSDK_CRUD_CONTROLS_UPDATE),
        )
        values["crud_controls_update"] = bool(crud_controls_update)

        crud_controls_update_list = get_from_dict_or_env(
            values,
            "crud_controls_update_list",
            "ANYSDK_CRUD_CONTROLS_UPDATE_LIST",
            default=ANYSDK_CRUD_CONTROLS_UPDATE_LIST,
        )
        values["crud_controls_update_list"] = crud_controls_update_list.split(",")

        crud_controls_delete = get_from_dict_or_env(
            values,
            "crud_controls_delete",
            "ANYSDK_CRUD_CONTROLS_DELETE",
            default=bool(ANYSDK_CRUD_CONTROLS_DELETE),
        )
        values["crud_controls_delete"] = bool(crud_controls_delete)

        crud_controls_delete_list = get_from_dict_or_env(
            values,
            "crud_controls_delete_list",
            "ANYSDK_CRUD_CONTROLS_DELETE_LIST",
            default=ANYSDK_CRUD_CONTROLS_DELETE_LIST,
        )
        values["crud_controls_delete_list"] = crud_controls_delete_list.split(",")

        return values

    def __init__(self, **data: dict) -> None:
        super().__init__(**data)
        self.operations = self._build_operations()

    def _build_operations(self) -> list:
        operations = []
        sdk_functions = [
            func
            for func in dir(self.client)
            if callable(getattr(self.client, func)) and not func.startswith("_")
        ]

        for func_name in sdk_functions:
            func = getattr(self.client, func_name)
            operation = {
                "mode": func_name,
                "name": func.__name__.replace("_", " ").title(),
                "description": func.__doc__,
            }

            if self.crud_controls_create and any(
                word.lower() in func_name.lower()
                for word in self.crud_controls_create_list
            ):
                operations.append(operation)

            if self.crud_controls_read and any(
                word.lower() in func_name.lower()
                for word in self.crud_controls_read_list
            ):
                operations.append(operation)

            if self.crud_controls_update and any(
                word.lower() in func_name.lower()
                for word in self.crud_controls_update_list
            ):
                operations.append(operation)

            if self.crud_controls_delete and any(
                word.lower() in func_name.lower()
                for word in self.crud_controls_delete_list
            ):
                operations.append(operation)

        return operations

    def run(self, mode: str, query: str) -> str:
        try:
            params = json.loads(query)
            func = getattr(self.client, mode)
            result = func(**params)
            return json.dumps(result)
        except AttributeError:
            return f"Invalid mode: {mode}"
