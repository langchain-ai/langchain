from typing import Dict, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.gmail.base import GmailBaseTool


class GetThreadSchema(BaseModel):
    """Input for GetMessageTool."""

    # From https://support.google.com/mail/answer/7190?hl=en
    thread_id: str = Field(
        ...,
        description="The thread ID.",
    )


class GmailGetThread(GmailBaseTool):
    """Tool that gets a thread by ID from Gmail."""

    name: str = "get_gmail_thread"
    description: str = (
        "Use this tool to search for email messages."
        " The input must be a valid Gmail query."
        " The output is a JSON list of messages."
    )
    args_schema: Type[GetThreadSchema] = GetThreadSchema

    def _run(
        self,
        thread_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Run the tool."""
        query = self.api_resource.users().threads().get(userId="me", id=thread_id)
        thread_data = query.execute()
        if not isinstance(thread_data, dict):
            raise ValueError("The output of the query must be a list.")
        messages = thread_data["messages"]
        thread_data["messages"] = []
        keys_to_keep = ["id", "snippet", "snippet"]
        # TODO: Parse body.
        for message in messages:
            thread_data["messages"].append(
                {k: message[k] for k in keys_to_keep if k in message}
            )
        return thread_data
