"""Tool for agent to sleep."""
from asyncio import sleep as asleep
from time import sleep
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


class SleepInput(BaseModel):
    """Input for CopyFileTool."""

    sleep_time: int = Field(..., description="Time to sleep in seconds")


class SleepToolSchema(BaseModel):
    """Input schema for SleepTool."""

    sleep_time: str = Field('Time you want to execute the sleep')

class SleepTool(BaseTool):
    """Tool that adds the capability to sleep."""

    name: str = "sleep"
    args_schema: Type[BaseModel] = SleepInput
    description: str = "Make agent sleep for a specified number of seconds."
    args_schema: Type[SleepToolSchema] = SleepToolSchema

    def _run(
        self,
        sleep_time: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Sleep tool."""
        sleep(sleep_time)
        return f"Agent slept for {sleep_time} seconds."

    async def _arun(
        self,
        sleep_time: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the sleep tool asynchronously."""
        await asleep(sleep_time)
        return f"Agent slept for {sleep_time} seconds."
