import base64
import email
from typing import Dict, Optional, Sequence, Type

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


def _decode_payload(
    payload: bytes,
    charset: Optional[str],
    fallback_encodings: Sequence[str],
    *,
    allow_utf8_replace: bool = True
) -> str:
    """Decode a payload using a list of encodings.

    Args:
        payload: The payload to decode.
        charset: The charset to use for decoding. If None, will try utf-8 first.
        fallback_encodings: A list of encodings to try in order.
        allow_utf8_replace: The error handling scheme to use for decoding. For example,
            "replace" will replace unknown characters with a question mark.
            Decodes using utf-8 with replacement for unknown characters.
            This is used as a fallback when all other encodings fail.

    Returns:
        The decoded payload if decoding was successful, otherwise will attempt
        to decode using utf-8 with replacement for unknown characters.
    """
    if charset:
        all_encodings = [charset] + list(fallback_encodings)
    else:
        all_encodings = list(fallback_encodings)

    for encoding in all_encodings:
        try:
            return payload.decode(encoding=encoding)
        except UnicodeDecodeError:
            pass

    # Default to utf-8 with replacement for unknown characters
    if allow_utf8_replace:
        return payload.decode("utf-8", errors="ignore")
    else:
        raise UnicodeDecodeError(
            "Unable to decode payload using any of the specified encodings."
        )


class GmailGetMessage(GmailBaseTool):
    """Tool that gets a message by ID from Gmail."""

    name: str = "get_gmail_message"
    description: str = (
        "Use this tool to fetch an email by message ID."
        " Returns the thread ID, snippet, body, subject, and sender."
    )
    args_schema: Type[SearchArgsSchema] = SearchArgsSchema
    fallback_encodings: Sequence[str] = (
        "utf-8",
        "latin1",
        "iso-8859-1",
        "cp1252",
    )
    """Encodings to try when decoding the email body if the charset is not specified.
    
    The decoder will first try the charset specified in the email. If that fails
    it'll try the encodings specified in this tuple.
    
    Finally, if all else fails, it'll try utf-8 with replacement for unknown characters.
    """
    allow_utf8_replace: bool = True
    """If true, and decoding using fallback encodings fails, try utf-8 with replacement 
    for unknown characters."""

    def _run(
        self,
        message_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Run the tool."""

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
                # Get the content type of the current email part
                # This could be "text/plain" or "text/html" or in theory,
                # any other MIME type.
                # If the email is not plain text, we ignore it.
                ctype = part.get_content_type()
                # Get the content disposition
                # The content disposition indicates whether the part is an inline
                # part or whether it is an attachment.
                cdispo = str(part.get("Content-Disposition"))
                # Currently this code does not handle non text/plain content types
                # and attachments.
                if ctype == "text/plain" and "attachment" not in cdispo:
                    charset = part.get_content_charset()  # Extract charset
                    message_body = _decode_payload(
                        part.get_payload(decode=True),
                        charset,
                        # Try the current charset, then utf-8, then latin1, then cp1252
                        self.fallback_encodings,
                        allow_utf8_replace=self.allow_utf8_replace,
                    )
                    break
        else:
            # The email message is not multipart
            # Get the charset and payload
            charset = email_msg.get_content_charset()
            message_body = _decode_payload(
                email_msg.get_payload(decode=True),
                charset,
                self.fallback_encodings,
                allow_utf8_replace=self.allow_utf8_replace,
            )

        body = clean_email_body(message_body)

        return {
            "id": message_id,
            "threadId": message_data["threadId"],
            "snippet": message_data["snippet"],
            "body": body,
            "subject": subject,
            "sender": sender,
        }
