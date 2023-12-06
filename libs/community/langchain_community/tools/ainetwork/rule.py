import builtins
import json
from typing import Optional, Type

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.ainetwork.base import AINBaseTool, OperationType


class RuleSchema(BaseModel):
    """Schema for owner operations."""

    type: OperationType = Field(...)
    path: str = Field(..., description="Path on the blockchain where the rule applies")
    eval: Optional[str] = Field(None, description="eval string to determine permission")


class AINRuleOps(AINBaseTool):
    """Tool for owner operations."""

    name: str = "AINruleOps"
    description: str = """
Covers the write `rule` for the AINetwork Blockchain database. The SET type specifies write permissions using the `eval` variable as a JavaScript eval string.
In order to AINvalueOps with SET at the path, the execution result of the `eval` string must be true.

## Path Rules
1. Allowed characters for directory: `[a-zA-Z_0-9]`
2. Use `$<key>` for template variables as directory.

## Eval String Special Variables
- auth.addr: Address of the writer for the path
- newData: New data for the path
- data: Current data for the path
- currentTime: Time in seconds
- lastBlockNumber: Latest processed block number

## Eval String Functions
- getValue(<path>)
- getRule(<path>)
- getOwner(<path>)
- getFunction(<path>)
- evalRule(<path>, <value to set>, auth, currentTime)
- evalOwner(<path>, 'write_owner', auth)

## SET Example
- type: SET
- path: /apps/langchain_project_1/$from/$to/$img
- eval: auth.addr===$from&&!getValue('/apps/image_db/'+$img)

## GET Example
- type: GET
- path: /apps/langchain_project_1
"""  # noqa: E501
    args_schema: Type[BaseModel] = RuleSchema

    async def _arun(
        self,
        type: OperationType,
        path: str,
        eval: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            if type is OperationType.SET:
                if eval is None:
                    raise ValueError("'eval' is required for SET operation.")

                res = await self.interface.db.ref(path).setRule(
                    transactionInput=ValueOnlyTransactionInput(
                        value={".rule": {"write": eval}}
                    )
                )
            elif type is OperationType.GET:
                res = await self.interface.db.ref(path).getRule()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{builtins.type(e).__name__}: {str(e)}"
