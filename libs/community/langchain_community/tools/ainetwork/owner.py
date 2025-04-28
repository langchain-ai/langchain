import builtins
import json
from typing import List, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType


class RuleSchema(BaseModel):
    """Schema for owner operations."""

    type: OperationType = Field(...)
    path: str = Field(..., description="Blockchain reference path")
    address: Optional[Union[str, List[str]]] = Field(
        None, description="A single address or a list of addresses"
    )
    write_owner: Optional[bool] = Field(
        False, description="Authority to edit the `owner` property of the path"
    )
    write_rule: Optional[bool] = Field(
        False, description="Authority to edit `write rule` for the path"
    )
    write_function: Optional[bool] = Field(
        False, description="Authority to `set function` for the path"
    )
    branch_owner: Optional[bool] = Field(
        False, description="Authority to initialize `owner` of sub-paths"
    )


class AINOwnerOps(AINBaseTool):
    """Tool for owner operations."""

    name: str = "AINownerOps"
    description: str = """
Rules for `owner` in AINetwork Blockchain database.
An address set as `owner` can modify permissions according to its granted authorities

## Path Rule
- (/[a-zA-Z_0-9]+)+
- Permission checks ascend from the most specific (child) path to broader (parent) paths until an `owner` is located.

## Address Rules
- 0x[0-9a-fA-F]{40}: 40-digit hexadecimal address
- *: All addresses permitted
- Defaults to the current session's address

## SET
- `SET` alters permissions for specific addresses, while other addresses remain unaffected.
- When removing an address of `owner`, set all authorities for that address to false.
- message `write_owner permission evaluated false` if fail

### Example
- type: SET
- path: /apps/langchain
- address: [<address 1>, <address 2>]
- write_owner: True
- write_rule: True
- write_function: True
- branch_owner: True

## GET
- Provides all addresses with `owner` permissions and their authorities in the path.

### Example
- type: GET
- path: /apps/langchain
"""  # noqa: E501
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(
        self,
        type: OperationType,
        path: str,
        address: Optional[Union[str, List[str]]] = None,
        write_owner: Optional[bool] = None,
        write_rule: Optional[bool] = None,
        write_function: Optional[bool] = None,
        branch_owner: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            if type is OperationType.SET:
                if address is None:
                    address = self.interface.wallet.defaultAccount.address
                if isinstance(address, str):
                    address = [address]
                res = await self.interface.db.ref(path).setOwner(
                    transactionInput=ValueOnlyTransactionInput(
                        value={
                            ".owner": {
                                "owners": {
                                    address: {
                                        "write_owner": write_owner or False,
                                        "write_rule": write_rule or False,
                                        "write_function": write_function or False,
                                        "branch_owner": branch_owner or False,
                                    }
                                    for address in address
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
