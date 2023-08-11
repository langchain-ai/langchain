import builtins
import json
from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool, OperationType


class RuleSchema(BaseModel):
    type: OperationType = Field(...)
    path: str = Field(..., description="Blockchain reference path")
    address: Optional[str] = Field(
        "*", description="eval string to determine permission"
    )
    write_owner: Optional[bool] = Field(
        False, description="eval string to determine permission"
    )
    write_rule: Optional[bool] = Field(
        False, description="eval string to determine permission"
    )
    write_function: Optional[bool] = Field(
        False, description="eval string to determine permission"
    )
    branch_owner: Optional[bool] = Field(
        False, description="eval string to determine permission"
    )


class AINOwnerOps(AINBaseTool):
    name: str = "AINownerOps"
    description: str = """
Rules for ownership in AINetwork Blockchain database.

## Path Specific Rules
- Valid characters: `[a-zA-Z_0-9]`

## Address Rules
- 0x[0-9a-fA-F]{64}: 64-digit hexadecimal public address
- *: Allows all address

## Permission Types
- write_owner: Edit ownership of the path
- write_rule: Edit rules for the path
- write_function: Set function for the path
- branch_owner: Inherits ownership for sub-paths

## SET Example
- type: SET
- path: /apps/afan
- address: *
- write_owner: True
- write_rule: True
- write_function: True
- branch_owner: True

## GET Example
- type: GET
- path: /apps/afan
"""
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(
        self,
        type: OperationType,
        path: str,
        address: Optional[str],
        write_owner: Optional[bool],
        write_rule: Optional[bool],
        write_function: Optional[bool],
        branch_owner: Optional[bool],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            if type is OperationType.SET:
                res = await self.interface.db.ref(path).setOwner(
                    transactionInput=ValueOnlyTransactionInput(
                        value={
                            ".owner": {
                                "owners": {
                                    address: {
                                        "write_owner": write_owner,
                                        "write_rule": write_rule,
                                        "write_function": write_function,
                                        "branch_owner": branch_owner,
                                    }
                                }
                            }
                        }
                    )
                )
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getOwner()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{builtins.type(e).__name__}: {str(e)}"
