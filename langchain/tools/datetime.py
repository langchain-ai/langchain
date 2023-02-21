"""Tool to get the current date and time."""
from datetime import datetime

from langchain.tools.base import BaseTool


class DateTimeTool(BaseTool):
    """Tool to get the current date and time."""

    name = "DateTime"
    description = (
        "A method to get the CURRENT date and time. "
        "The input to this should be an empty string."
    )

    def _run(self, query: str) -> str:
        """Use the DateTime tool."""
        now = datetime.now()
        dt_string = now.strftime("%A, %m/%d/%Y %H:%M:%S")
        return f"The current date and time is: {dt_string}"

    async def _arun(self, query: str) -> str:
        """Use the DateTime tool asynchronously."""
        raise NotImplementedError("DateTimeTool does not support async")
