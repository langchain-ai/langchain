import builtins
import json
from typing import Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType


class ValueSchema(BaseModel):
    """Schema for value operations."""

    type: OperationType = Field(...)
    path: str = Field(..., description="Blockchain reference path")
    value: Optional[Union[int, str, float, dict]] = Field(
        None, description="Value to be set at the path"
    )


class AINValueOps(AINBaseTool):
    """Tool for value operations."""

    name: str = "AINvalueOps"
    description: str = """
Covers the read and write value for the AINetwork Blockchain database.

## SET
- Set a value at a given path

### Example
- type: SET
- path: /apps/langchain_test_1/object
- value: {1: 2, "34": 56}

## GET
- Retrieve a value at a given path

### Example
- type: GET
- path: /apps/langchain_test_1/DB

## Special paths
- `/accounts/<address>/balance`: Account balance
- `/accounts/<address>/nonce`: Account nonce
- `/apps`: Applications
- `/consensus`: Consensus
- `/checkin`: Check-in
- `/deposit/<service id>/<address>/<deposit id>`: Deposit
- `/deposit_accounts/<service id>/<address>/<account id>`: Deposit accounts
- `/escrow`: Escrow
- `/payments`: Payment
- `/sharding`: Sharding
- `/token/name`: Token name
- `/token/symbol`: Token symbol
- `/token/total_supply`: Token total supply
- `/transfer/<address from>/<address to>/<key>/value`: Transfer
- `/withdraw/<service id>/<address>/<withdraw id>`: Withdraw
"""
    args_schema: Type[BaseModel] = ValueSchema

    async def _arun(
        self,
        type: OperationType,
        path: str,
        value: Optional[Union[int, str, float, dict]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            if type is OperationType.SET:
                if value is None:
                    raise ValueError("'value' is required for SET operation.")

                res = await self.interface.db.ref(path).setValue(
                    transactionInput=ValueOnlyTransactionInput(value=value)
                )
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getValue()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{builtins.type(e).__name__}: {str(e)}"
