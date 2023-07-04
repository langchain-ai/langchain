from typing import List, Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.office365.base import O365BaseTool


class SendMessageSchema(BaseModel):
    body: str = Field(
        ...,
        description="The message body to be sent.",
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


class O365SendMessage(O365BaseTool):
    name: str = "send_email"
    description: str = (
        "Use this tool to send an email with the provided message fields."
    )
    args_schema: Type[SendMessageSchema] = SendMessageSchema

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
            message.bcc.add(cc)

        message.send()

        output = "Message sent: " + str(message)
        return output

    async def _arun(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
