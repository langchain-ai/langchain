"""Tool for agent to sleep."""
from asyncio import sleep as asleep
from time import sleep
from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


class SleepInput(BaseModel):
    """Input for CopyFileTool."""

    sleep_time: int = Field(..., description="Time to sleep in seconds")


class SleepTool(BaseTool):
    """Tool that adds the capability to sleep."""

    name = "sleep"
    args_schema: Type[BaseModel] = SleepInput
    description = "Make agent sleep for a specified number of seconds."

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
