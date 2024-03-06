from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

response = (
    "Create a final answer that says if they "
    "have any questions about movies or actors"
)


class SmalltalkInput(BaseModel):
    query: Optional[str] = Field(description="user query")


class SmalltalkTool(BaseTool):
    name = "Smalltalk"
    description = "useful for when user greets you or wants to smalltalk"
    args_schema: Type[BaseModel] = SmalltalkInput

    def _run(
        self,
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return response

    async def _arun(
        self,
        query: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return response
