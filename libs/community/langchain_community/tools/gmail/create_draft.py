import base64
from email.message import EmailMessage
from typing import List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.gmail.base import GmailBaseTool


class CreateDraftSchema(BaseModel):
    """Input for CreateDraftTool."""

    message: str = Field(
        ...,
        description="The message to include in the draft.",
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


class GmailCreateDraft(GmailBaseTool):  # type: ignore[override, override]
    """Tool that creates a draft email for Gmail."""

    name: str = "create_gmail_draft"
    description: str = (
        "Use this tool to create a draft email with the provided message fields."
    )
    args_schema: Type[CreateDraftSchema] = CreateDraftSchema

    def _prepare_draft_message(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
    ) -> dict:
        draft_message = EmailMessage()
        draft_message.set_content(message)

        draft_message["To"] = ", ".join(to)
        draft_message["Subject"] = subject
        if cc is not None:
            draft_message["Cc"] = ", ".join(cc)

        if bcc is not None:
            draft_message["Bcc"] = ", ".join(bcc)

        encoded_message = base64.urlsafe_b64encode(draft_message.as_bytes()).decode()
        return {"message": {"raw": encoded_message}}

    def _run(
        self,
        message: str,
        to: List[str],
        subject: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            create_message = self._prepare_draft_message(message, to, subject, cc, bcc)
            draft = (
                self.api_resource.users()
                .drafts()
                .create(userId="me", body=create_message)
                .execute()
            )
            output = f"Draft created. Draft Id: {draft['id']}"
            return output
        except Exception as e:
            raise Exception(f"An error occurred: {e}")
