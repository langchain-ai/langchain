import builtins
import json
from enum import Enum
from typing import List, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.ainetwork.base import AINBaseTool


class AppOperationType(str, Enum):
    """Type of app operation as enumerator."""

    SET_ADMIN = "SET_ADMIN"
    GET_CONFIG = "GET_CONFIG"


class AppSchema(BaseModel):
    """Schema for app operations."""

    type: AppOperationType = Field(...)
    appName: str = Field(..., description="Name of the application on the blockchain")
    address: Optional[Union[str, List[str]]] = Field(
        None,
        description=(
            "A single address or a list of addresses. Default: current session's "
            "address"
        ),
    )


class AINAppOps(AINBaseTool):
    """Tool for app operations."""

    name: str = "AINappOps"
    description: str = """
Create an app in the AINetwork Blockchain database by creating the /apps/<appName> path.
An address set as `admin` can grant `owner` rights to other addresses (refer to `AINownerOps` for more details).
Also, `admin` is initialized to have all `owner` permissions and `rule` allowed for that path.

## appName Rule
- [a-z_0-9]+

## address Rules
- 0x[0-9a-fA-F]{40}
- Defaults to the current session's address
- Multiple addresses can be specified if needed

## SET_ADMIN Example 1
- type: SET_ADMIN
- appName: ain_project

### Result:
1. Path /apps/ain_project created.
2. Current session's address registered as admin.

## SET_ADMIN Example 2
- type: SET_ADMIN
- appName: test_project
- address: [<address1>, <address2>]

### Result:
1. Path /apps/test_project created.
2. <address1> and <address2> registered as admin.

"""  # noqa: E501
    args_schema: Type[BaseModel] = AppSchema

    async def _arun(
        self,
        type: AppOperationType,
        appName: str,
        address: Optional[Union[str, List[str]]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput
        from ain.utils import getTimestamp

        try:
            if type is AppOperationType.SET_ADMIN:
                if address is None:
                    address = self.interface.wallet.defaultAccount.address
                if isinstance(address, str):
                    address = [address]

                res = await self.interface.db.ref(
                    f"/manage_app/{appName}/create/{getTimestamp()}"
                ).setValue(
                    transactionInput=ValueOnlyTransactionInput(
                        value={"admin": {address: True for address in address}}
                    )
                )
            elif type is AppOperationType.GET_CONFIG:
                res = await self.interface.db.ref(
                    f"/manage_app/{appName}/config"
                ).getValue()
            else:
                raise ValueError(f"Unsupported 'type': {type}.")
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{builtins.type(e).__name__}: {str(e)}"
