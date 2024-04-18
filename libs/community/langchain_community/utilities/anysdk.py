import json
from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

ANYSDK_CRUD_CONTROLS_CREATE = False
ANYSDK_CRUD_CONTROLS_READ = True
ANYSDK_CRUD_CONTROLS_UPDATE = False
ANYSDK_CRUD_CONTROLS_DELETE = False

ANYSDK_CRUD_CONTROLS_CREATE_LIST = "create"
ANYSDK_CRUD_CONTROLS_READ_LIST = "get,read,list"
ANYSDK_CRUD_CONTROLS_UPDATE_LIST = "update,put,post"
ANYSDK_CRUD_CONTROLS_DELETE_LIST = "delete,destroy,remove"


class CrudControls(BaseModel):
    create: Optional[bool] = None
    create_list: Optional[str] = None
    read: Optional[bool] = None
    read_list: Optional[str] = None
    update: Optional[bool] = None
    update_list: Optional[str] = None
    delete: Optional[bool] = None
    delete_list: Optional[str] = None

    @root_validator
    def validate_environment(cls, values: dict) -> dict:
        create = get_from_dict_or_env(
            values,
            "create",
            "ANYSDK_CRUD_CONTROLS_CREATE",
            default=ANYSDK_CRUD_CONTROLS_CREATE,
        )
        values["create"] = bool(create)

        create_list: str = get_from_dict_or_env(
            values,
            "create_list",
            "ANYSDK_CRUD_CONTROLS_CREATE_LIST",
            default=ANYSDK_CRUD_CONTROLS_CREATE_LIST,
        )
        if create_list:
            values["create_list"] = create_list.split(",")
        else:
            values["create_list"] = []

        read = get_from_dict_or_env(
            values,
            "read",
            "ANYSDK_CRUD_CONTROLS_READ",
            default=ANYSDK_CRUD_CONTROLS_READ,
        )
        values["read"] = bool(read)

        read_list = get_from_dict_or_env(
            values,
            "read_list",
            "ANYSDK_CRUD_CONTROLS_READ_LIST",
            default=ANYSDK_CRUD_CONTROLS_READ_LIST,
        )
        values["read_list"] = read_list.split(",")

        update = get_from_dict_or_env(
            values,
            "update",
            "ANYSDK_CRUD_CONTROLS_UPDATE",
            default=ANYSDK_CRUD_CONTROLS_UPDATE,
        )
        values["update"] = bool(update)

        update_list = get_from_dict_or_env(
            values,
            "update_list",
            "ANYSDK_CRUD_CONTROLS_UPDATE_LIST",
            default=ANYSDK_CRUD_CONTROLS_UPDATE_LIST,
        )
        values["update_list"] = update_list.split(",")

        delete = get_from_dict_or_env(
            values,
            "delete",
            "ANYSDK_CRUD_CONTROLS_DELETE",
            default=ANYSDK_CRUD_CONTROLS_DELETE,
        )
        values["delete"] = bool(delete)

        delete_list = get_from_dict_or_env(
            values,
            "delete_list",
            "ANYSDK_CRUD_CONTROLS_DELETE_LIST",
            default=ANYSDK_CRUD_CONTROLS_DELETE_LIST,
        )
        values["delete_list"] = delete_list.split(",")

        return values


class AnySdkWrapper(BaseModel):
    client: Any
    operations: List[Dict] = []
    crud_controls: Optional[CrudControls] = None

    class Config:
        extra = Extra.forbid

    def __init__(self, **data: dict) -> None:
        super().__init__(**data)
        self.operations = self._build_operations()

    def _build_operations(self) -> list:
        operations = []
        sdk_functions = [
            func
            for func in dir(self.client["client"])
            if (
                callable(getattr(self.client["client"], func))
                and not func.startswith("_")
            )
        ]

        for func_name in sdk_functions:
            func = getattr(self.client["client"], func_name)
            operation = {
                "mode": func_name,
                "name": func.__name__.replace("_", " ").title(),
                "description": func.__doc__,
            }

            if self.crud_controls.create:
                if self.crud_controls.create_list is not None and any(
                    word.lower() in func_name.lower()
                    for word in self.crud_controls.create_list
                ):
                    operations.append(operation)

            if self.crud_controls.read:
                if self.crud_controls.read_list is not None and any(
                    word.lower() in func_name.lower()
                    for word in self.crud_controls.read_list
                ):
                    operations.append(operation)

            if self.crud_controls.update:
                if self.crud_controls.update_list is not None and any(
                    word.lower() in func_name.lower()
                    for word in self.crud_controls.update_list
                ):
                    operations.append(operation)

            if self.crud_controls.delete:
                if self.crud_controls.delete_list is not None and any(
                    word.lower() in func_name.lower()
                    for word in self.crud_controls.delete_list
                ):
                    operations.append(operation)

        return operations

    def run(self, mode: str, query: str) -> str:
        try:
            params = json.loads(query)
            func = getattr(self.client["client"], mode)
            result = func(**params)
            return json.dumps(result)
        except AttributeError:
            return f"Invalid mode: {mode}"
