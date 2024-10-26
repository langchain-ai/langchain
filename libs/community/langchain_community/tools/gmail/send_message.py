"""Send Gmail messages."""

import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.gmail.base import GmailBaseTool


class SendMessageSchema(BaseModel):
    """Input for SendMessageTool."""

    message: str = Field(
        ...,
        description="The message to send.",
    )
    to: Union[str, List[str]] = Field(
        ...,
        description="The list of recipients.",
    )
    subject: str = Field(
        ...,
        description="The subject of the message.",
    )
    cc: Optional[Union[str, List[str]]] = Field(
        None,
        description="The list of CC recipients.",
    )
    bcc: Optional[Union[str, List[str]]] = Field(
        None,
        description="The list of BCC recipients.",
    )


class GmailSendMessage(GmailBaseTool):  # type: ignore[override, override]
    """Tool that sends a message to Gmail."""

    name: str = "send_gmail_message"
    description: str = (
        "Use this tool to send email messages." " The input is the message, recipients"
    )
    args_schema: Type[SendMessageSchema] = SendMessageSchema

    def _prepare_message(
        self,
        message: str,
        to: Union[str, List[str]],
        subject: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Create a message for an email."""
        mime_message = MIMEMultipart()
        mime_message.attach(MIMEText(message, "html"))

        mime_message["To"] = ", ".join(to if isinstance(to, list) else [to])
        mime_message["Subject"] = subject
        if cc is not None:
            mime_message["Cc"] = ", ".join(cc if isinstance(cc, list) else [cc])

        if bcc is not None:
            mime_message["Bcc"] = ", ".join(bcc if isinstance(bcc, list) else [bcc])

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        return {"raw": encoded_message}

    def _run(
        self,
        message: str,
        to: Union[str, List[str]],
        subject: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
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
