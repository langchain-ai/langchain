"""Tool for agent to sleep."""
from typing import Type
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field
from time import sleep
from asyncio import sleep as asleep


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
    ) -> str:
        """Use the Sleep tool."""
        sleep(sleep_time)
        return f"Agent slept for {sleep_time} seconds."

    async def _arun(
        self,
        sleep_time: int,
    ) -> str:
        """Use the sleep tool asynchronously."""
        await asleep(sleep_time)
        return f"Agent slept for {sleep_time} seconds."
