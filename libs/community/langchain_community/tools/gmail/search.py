import base64
import email
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.gmail.base import GmailBaseTool
from langchain_community.tools.gmail.utils import clean_email_body


class Resource(str, Enum):
    """Enumerator of Resources to search."""

    THREADS = "threads"
    MESSAGES = "messages"


class SearchArgsSchema(BaseModel):
    """Input for SearchGmailTool."""

    # From https://support.google.com/mail/answer/7190?hl=en
    query: str = Field(
        ...,
        description="The Gmail query. Example filters include from:sender,"
        " to:recipient, subject:subject, -filtered_term,"
        " in:folder, is:important|read|starred, after:year/mo/date, "
        "before:year/mo/date, label:label_name"
        ' "exact phrase".'
        " Search newer/older than using d (day), m (month), and y (year): "
        "newer_than:2d, older_than:1y."
        " Attachments with extension example: filename:pdf. Multiple term"
        " matching example: from:amy OR from:david.",
    )
    resource: Resource = Field(
        default=Resource.MESSAGES,
        description="Whether to search for threads or messages.",
    )
    max_results: int = Field(
        default=10,
        description="The maximum number of results to return.",
    )


class GmailSearch(GmailBaseTool):
    """Tool that searches for messages or threads in Gmail."""

    name: str = "search_gmail"
    description: str = (
        "Use this tool to search for email messages or threads."
        " The input must be a valid Gmail query."
        " The output is a JSON list of the requested resource."
    )
    args_schema: Type[SearchArgsSchema] = SearchArgsSchema

    def _parse_threads(self, threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Add the thread message snippets to the thread results
        results = []
        for thread in threads:
            thread_id = thread["id"]
            thread_data = (
                self.api_resource.users()
                .threads()
                .get(userId="me", id=thread_id)
                .execute()
            )
            messages = thread_data["messages"]
            thread["messages"] = []
            for message in messages:
                snippet = message["snippet"]
                thread["messages"].append({"snippet": snippet, "id": message["id"]})
            results.append(thread)

        return results

    def _parse_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for message in messages:
            message_id = message["id"]
            message_data = (
                self.api_resource.users()
                .messages()
                .get(userId="me", format="raw", id=message_id)
                .execute()
            )

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
                        try:
                            message_body = part.get_payload(decode=True).decode("utf-8")  # type: ignore[union-attr]
                        except UnicodeDecodeError:
                            message_body = part.get_payload(decode=True).decode(  # type: ignore[union-attr]
                                "latin-1"
                            )
                        break
            else:
                message_body = email_msg.get_payload(decode=True).decode("utf-8")  # type: ignore[union-attr]

            body = clean_email_body(message_body)

            results.append(
                {
                    "id": message["id"],
                    "threadId": message_data["threadId"],
                    "snippet": message_data["snippet"],
                    "body": body,
                    "subject": subject,
                    "sender": sender,
                    "from": email_msg["From"],
                    "date": email_msg["Date"],
                    "to": email_msg["To"],
                    "cc": email_msg["Cc"],
                }
            )
        return results

    def _run(
        self,
        query: str,
        resource: Resource = Resource.MESSAGES,
        max_results: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool."""
        results = (
            self.api_resource.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
            .get(resource.value, [])
        )
        if resource == Resource.THREADS:
            return self._parse_threads(results)
        elif resource == Resource.MESSAGES:
            return self._parse_messages(results)
        else:
            raise NotImplementedError(f"Resource of type {resource} not implemented.")
