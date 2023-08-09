import json
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.ainetwork.base import AINBaseTool
from pydantic import BaseModel, Field


class GetValueSchema(BaseModel):
    path: str = Field(..., description="Blockchain reference path")


class AINGetValue(AINBaseTool):
    name: str = "AINgetvalue"
    description: str = "Retrieve a value from a given path on the AIN blockchain"
    args_schema: Type[BaseModel] = GetValueSchema

    async def _arun(
        self,
        path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        value = await self.interface.db.ref(path).getValue()
        return json.dumps(value, ensure_ascii=False)
