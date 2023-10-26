import base64
import email
from typing import Dict, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.gmail.base import GmailBaseTool
from langchain.tools.gmail.utils import clean_email_body


class SearchArgsSchema(BaseModel):
    """Input for GetMessageTool."""

    message_id: str = Field(
        ...,
        description="The unique ID of the email message, retrieved from a search.",
    )


class GmailGetMessage(GmailBaseTool):
    """Tool that gets a message by ID from Gmail."""

    name: str = "get_gmail_message"
    description: str = (
        "Use this tool to fetch an email by message ID."
        " Returns the thread ID, snippet, body, subject, and sender."
    )
    args_schema: Type[SearchArgsSchema] = SearchArgsSchema

    def _run(
        self,
        message_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Run the tool."""

        def decode_payload(payload, charset=None):
            encodings_to_try = [
                "utf-8",
                "latin1",
                "iso-8859-1",
                "cp1252",
            ]  # Add more if needed

            if charset:
                encodings_to_try.insert(0, charset)

            for encoding in encodings_to_try:
                try:
                    return payload.decode(encoding)
                except Exception:
                    pass

            # Default to utf-8 with replacement for unknown characters
            return payload.decode("utf-8", "replace")

        query = (
            self.api_resource.users()
            .messages()
            .get(userId="me", format="raw", id=message_id)
        )
        message_data = query.execute()
        raw_message = base64.urlsafe_b64decode(message_data["raw"])

        email_msg = email.message_from_bytes(raw_message)

        subject = email_msg["Subject"]
        sender = email_msg["From"]

        message_body = ""
        if email_msg.is_multipart():
            for part in email_msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdispo:
                    charset = part.get_content_charset()  # Extract charset
                    message_body = decode_payload(
                        part.get_payload(decode=True), charset
                    )
                    break
        else:
            charset = email_msg.get_content_charset()
            message_body = decode_payload(email_msg.get_payload(decode=True), charset)

        body = clean_email_body(message_body)

        return {
            "id": message_id,
            "threadId": message_data["threadId"],
            "snippet": message_data["snippet"],
            "body": body,
            "subject": subject,
            "sender": sender,
        }
