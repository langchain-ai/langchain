import json
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool


class SetFunctionSchema(BaseModel):
    path: str = Field(..., description="Blockchain reference path")
    value: str = Field(..., description="Function value to be set at the path")


class AINSetFunction(AINBaseTool):
    name: str = "AINsetfunction"
    description: str = "Sets a function at a given path on the AIN blockchain"
    args_schema: Type[BaseModel] = SetFunctionSchema

    async def _arun(
        self,
        path: str,
        value: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from ain.types import ValueOnlyTransactionInput

        try:
            res = await self.interface.db.ref(path).setFunction(
                transactionInput=ValueOnlyTransactionInput(value=value)
            )
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return f"{type(e).__name__}: {str(e)}"
