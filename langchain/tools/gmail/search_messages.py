import base64
import email
import json
from typing import Any, Callable, Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.gmail.base import GmailBaseTool
from langchain.tools.gmail.utils import clean_email_body


QUERY_PROMPT = """The current date is {date}.

Convert the following input into a Gmail search query. Here are the search query specificaions:

Specify the sender
Example: from:amy

Specify a recipient
Example: to:david

Words in the subject line
Example: subject:dinner

Messages that match multiple terms
Example: from:amy OR from:david
Example: {{from:amy from:david}}

Remove messages from your results
Example: dinner -movie

Find messages with words near each other. Use the number to say how many words apart the words can be
Add quotes to find messages in which the word you put first stays first.
Example: holiday AROUND 10 vacation
Example: "secret AROUND 25 birthday"

Messages that have a certain label
Example: label:friends


Attachments with a certain name or file type
Example: filename:pdf

Search for an exact word or phrase
Example: "dinner and movie tonight"

Group multiple search terms together
Example: subject:(dinner movie)

Messages in any folder, including Spam and Trash
Example: in:anywhere movie

Search for messages that are marked as important
Example: is:important

Starred, snoozed, unread, or read messages
Example: is:read is:starred

Search for messages sent during a certain time period
Example: after:2004/04/16
Example: after:04/16/2004
Example: before:2004/04/18
Example: before:04/18/2004

Search for messages older or newer than a time period using d (day), m (month), and y (year)
Example: newer_than:2d

Input: {input}
Query:"""


# search_messages = Tool(
#     name="search_messages",
#     description="Use this tool to search for email messages. The input must be a valid Gmail query. The output is a JSON list of messages.",
#     func=self.search_messages,
#     return_direct=self.tools_return_direct,
# )

# From https://support.google.com/mail/answer/7190?hl=en


class GmailSearchMessages(GmailBaseTool):
    name: str = "search_gmail_messages"
    description: str = (
        "Use this tool to search for email messages."
        " The input must be a valid Gmail query."
        " The output is a JSON list of messages.",
    )
    query_mapper: Optional[Callable[[str], str]] = None

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Run the tool."""
        messages = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=10)
            .execute()
            .get("messages", [])
        )

        results = []
        for message in messages:
            message_id = message["id"]
            message_data = (
                self.service.users()
                .messages()
                .get(userId="me", format="raw", id=message_id)
                .execute()
            )

            raw_message = base64.urlsafe_b64decode(message_data["raw"])

            email_msg = email.message_from_bytes(raw_message)

            subject = email_msg["Subject"]
            sender = email_msg["From"]

            message_body = email_msg.get_payload()

            body = clean_email_body(message_body)

            results.append(
                {
                    "id": message["id"],
                    "threadId": message_data["threadId"],
                    "snippet": message_data["snippet"],
                    "body": body,
                    "subject": subject,
                    "sender": sender,
                }
            )

        return json.dumps(results)

    def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Run the tool."""
        raise NotImplementedError
