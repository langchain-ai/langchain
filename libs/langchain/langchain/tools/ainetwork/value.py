import json
from typing import Optional, Type, Union

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool, OperationType


class ValueSchema(BaseModel):
    type: OperationType = Field(...)
    path: str = Field(..., description="Blockchain reference path")
    value: Optional[Union[int, str, float, dict]] = Field(
        ..., description="Value to be set at the path"
    )


class AINValueOps(AINBaseTool):
    name: str = "AINvalueops"
    description: str = "On the AIN blockchain, use the SET operation to set a value at a given path. If you choose the GET operation, you can retrieve a value from that path. In this case, the value parameter is ignored."
    args_schema: Type[BaseModel] = ValueSchema

    async def _arun(
        self,
        path: str,
        value: Optional[Union[int, str, float, dict]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
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
            return f"{type(e).__name__}: {str(e)}"
