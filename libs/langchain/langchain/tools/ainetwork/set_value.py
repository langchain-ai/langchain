import json
from typing import Optional, Type, Union

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool
from pydantic import BaseModel, Field


class SetValueSchema(BaseModel):
    path: str = Field(..., description="Blockchain reference path")
    value: Union[int, str, float, dict] = Field(
        ..., description="Value to be set at the path"
    )


class AINSetValue(AINBaseTool):
    name: str = "AINSetValue"
    description: str = "Sets a value at a given path on the AIN blockchain"
    args_schema: Type[BaseModel] = SetValueSchema

    async def _arun(
        self,
        path: str,
        value: Union[int, str, float, dict],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from ain.errors import BlockchainError
        from ain.types import ValueOnlyTransactionInput

        try:
            res = await self.interface.db.ref(path).setValue(
                transactionInput=ValueOnlyTransactionInput(value=value)
            )
            return json.dumps(res, ensure_ascii=False)
        except BlockchainError as e:
            return f"Error: {e.message}"
