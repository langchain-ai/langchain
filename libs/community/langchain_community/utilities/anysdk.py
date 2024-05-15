from typing import Any, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.tools import BaseTool

from langchain_community.tools.anysdk.tool import AnySDKTool

ANYSDK_CRUD_CONTROLS_CREATE = False
ANYSDK_CRUD_CONTROLS_READ = True
ANYSDK_CRUD_CONTROLS_UPDATE = False
ANYSDK_CRUD_CONTROLS_DELETE = False

ANYSDK_CRUD_CONTROLS_CREATE_LIST = "create"
ANYSDK_CRUD_CONTROLS_READ_LIST = "get,read,list"
ANYSDK_CRUD_CONTROLS_UPDATE_LIST = "update,put,post"
ANYSDK_CRUD_CONTROLS_DELETE_LIST = "delete,destroy,remove"


class CrudControls(BaseModel):
    create: Optional[bool] = ANYSDK_CRUD_CONTROLS_CREATE
    create_list: Optional[str] = ANYSDK_CRUD_CONTROLS_CREATE_LIST
    read: Optional[bool] = ANYSDK_CRUD_CONTROLS_READ
    read_list: Optional[str] = ANYSDK_CRUD_CONTROLS_READ_LIST
    update: Optional[bool] = ANYSDK_CRUD_CONTROLS_UPDATE
    update_list: Optional[str] = ANYSDK_CRUD_CONTROLS_UPDATE_LIST
    delete: Optional[bool] = ANYSDK_CRUD_CONTROLS_DELETE
    delete_list: Optional[str] = ANYSDK_CRUD_CONTROLS_DELETE_LIST

    @root_validator
    def validate_environment(cls, values: dict) -> "CrudControls":
        create = values.get("create", ANYSDK_CRUD_CONTROLS_CREATE)
        values["create"] = create

        create_list = values.get("create_list", ANYSDK_CRUD_CONTROLS_CREATE_LIST)
        values["create_list"] = create_list.split(",")

        read = values.get("read", ANYSDK_CRUD_CONTROLS_READ)
        values["read"] = read

        read_list = values.get("read_list", ANYSDK_CRUD_CONTROLS_READ_LIST)
        values["read_list"] = read_list.split(",")

        update = values.get("update", ANYSDK_CRUD_CONTROLS_UPDATE)
        values["update"] = update

        update_list = values.get("update_list", ANYSDK_CRUD_CONTROLS_UPDATE_LIST)
        values["update_list"] = update_list.split(",")

        delete = values.get("delete", ANYSDK_CRUD_CONTROLS_DELETE)
        values["delete"] = delete

        delete_list = values.get("delete_list", ANYSDK_CRUD_CONTROLS_DELETE_LIST)
        values["delete_list"] = delete_list.split(",")

        return values  # type: ignore


class AnySdkWrapper(BaseModel):
    client: Any
    operations: List[AnySDKTool] = []
    crud_controls: CrudControls = CrudControls()

    class Config:
        extra = Extra.forbid

    def __init__(self, **data: dict) -> None:
        super().__init__(**data)
        self.operations = self._build_operations()

    def _build_operations(self) -> list:
        operations: List[BaseTool] = []
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
            operation = AnySDKTool(
                client=self.client, name=func_name, description=func.__doc__
            )
            if self.crud_controls:
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

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [op for op in self.operations]
