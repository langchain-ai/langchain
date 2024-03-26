"""Tool for the Dataherald Hosted API"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.dataherald import DataheraldAPIWrapper


class DataheraldTextToSQL(BaseTool):
    """Tool that queries using the Dataherald SDK."""

    name: str = "dataherald"
    description: str = (
        "A wrapper around Dataherald. "
        "Text to SQL. "
        "Input should be a prompt and an existing db_connection_id"
    )
    api_wrapper: DataheraldAPIWrapper

    def _run(
        self,
        prompt: str,
        db_connection_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Dataherald tool."""
        return self.api_wrapper.run(prompt, db_connection_id)
