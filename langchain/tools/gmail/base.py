from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.tools.gmail.utils import build_resource_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


# pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# prompt = QUERY_PROMPT.format(input=input, date=self._get_datetime())

#         query = self.llm(prompt)

#         print("Query: ", query)

#         service = build("gmail", "v1", credentials=self.credentials)

#         messages = (
#             service.users()
#             .messages()
#             .list(userId="me", q=query, maxResults=10)
#             .execute()
#             .get("messages", [])
#         )

#         results = []
#         for message in messages:
#             message_id = message["id"]
#             message_data = (
#                 self.service.users()
#                 .messages()
#                 .get(userId="me", format="raw", id=message_id)
#                 .execute()
#             )

#             raw_message = base64.urlsafe_b64decode(message_data["raw"])

#             email_msg = email.message_from_bytes(raw_message)

#             subject = email_msg["Subject"]
#             sender = email_msg["From"]

#             message_body = email_msg.get_payload()

#             body = self.clean_email_body(message_body)

#             results.append(
#                 {
#                     "id": message["id"],
#                     "threadId": message_data["threadId"],
#                     "snippet": message_data["snippet"],
#                     "body": body,
#                     "subject": subject,
#                     "sender": sender,
#                 }
#             )

#         return json.dumps(results)


class GmailBaseTool(BaseTool):
    name: str = "search_gmail_messages"
    description: str = (
        "Use this tool to search for email messages."
        " The input must be a valid Gmail query."
        " The output is a JSON list of messages.",
    )
    api_resource: Resource = Field(default_factory=build_resource_service)

    @classmethod
    def from_api_resource(cls, api_resource: Resource) -> "GmailBaseTool":
        return cls(service=api_resource)
