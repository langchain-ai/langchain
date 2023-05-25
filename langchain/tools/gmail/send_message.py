"""Send Gmail messages."""
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.gmail.base import GmailBaseTool


class SendMessageSchema(BaseModel):
    message: str = Field(
        ...,
        description="The message to send.",
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


class GmailSendMessage(GmailBaseTool):
    name: str = "send_gmail_message"
    description: str = (
        "Use this tool to send email messages." " The input is the message, recipents"
    )

    def _prepare_message(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a message for an email."""
        mime_message = MIMEMultipart()
        mime_message.attach(MIMEText(message, "html"))

        mime_message["To"] = ", ".join(to)
        mime_message["Subject"] = subject
        if cc is not None:
            mime_message["Cc"] = ", ".join(cc)

        if bcc is not None:
            mime_message["Bcc"] = ", ".join(bcc)

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        return {"raw": encoded_message}

    def _run(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool."""
        try:
            create_message = self._prepare_message(message, to, subject, cc=cc, bcc=bcc)
            send_message = (
                self.api_resource.users()
                .messages()
                .send(userId="me", body=create_message)
            )
            sent_message = send_message.execute()
            return f'Message sent. Message Id: {sent_message["id"]}'
        except Exception as error:
            raise Exception(f"An error occurred: {error}")

    async def _arun(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously."""
        raise NotImplementedError(f"The tool {self.name} does not support async yet.")
