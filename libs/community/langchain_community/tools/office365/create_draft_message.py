from typing import List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.office365.base import O365BaseTool


class CreateDraftMessageSchema(BaseModel):
    """Input for SendMessageTool."""

    body: str = Field(
        ...,
        description="The message body to include in the draft.",
    )
    to: List[str] = Field(
        ...,
        description="The list of recipients.",
    )
    subject: str = Field(
        ...,
        description="The subject of the message.",
    )
    cc: Optional[List[str]] = Field(
        None,
        description="The list of CC recipients.",
    )
    bcc: Optional[List[str]] = Field(
        None,
        description="The list of BCC recipients.",
    )


class O365CreateDraftMessage(O365BaseTool):
    """Tool for creating a draft email in Office 365."""

    name: str = "create_email_draft"
    description: str = (
        "Use this tool to create a draft email with the provided message fields."
    )
    args_schema: Type[CreateDraftMessageSchema] = CreateDraftMessageSchema

    def _run(
        self,
        body: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Get mailbox object
        mailbox = self.account.mailbox()
        message = mailbox.new_message()

        # Assign message values
        message.body = body
        message.subject = subject
        message.to.add(to)
        if cc is not None:
            message.cc.add(cc)
        if bcc is not None:
            message.bcc.add(bcc)

        message.save_draft()

        output = "Draft created: " + str(message)
        return output
