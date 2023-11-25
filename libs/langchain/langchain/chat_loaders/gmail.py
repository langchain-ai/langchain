import base64
import re
from typing import Any, Iterator

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage

from langchain.chat_loaders.base import BaseChatLoader


def _extract_email_content(msg: Any) -> HumanMessage:
    from_email = None
    for values in msg["payload"]["headers"]:
        name = values["name"]
        if name == "From":
            from_email = values["value"]
    if from_email is None:
        raise ValueError
    for part in msg["payload"]["parts"]:
        if part["mimeType"] == "text/plain":
            data = part["body"]["data"]
            data = base64.urlsafe_b64decode(data).decode("utf-8")
            # Regular expression to split the email body at the first
            # occurrence of a line that starts with "On ... wrote:"
            pattern = re.compile(r"\r\nOn .+(\r\n)*wrote:\r\n")
            # Split the email body and extract the first part
            newest_response = re.split(pattern, data)[0]
            message = HumanMessage(
                content=newest_response, additional_kwargs={"sender": from_email}
            )
            return message
    raise ValueError


def _get_message_data(service: Any, message: Any) -> ChatSession:
    msg = service.users().messages().get(userId="me", id=message["id"]).execute()
    message_content = _extract_email_content(msg)
    in_reply_to = None
    email_data = msg["payload"]["headers"]
    for values in email_data:
        name = values["name"]
        if name == "In-Reply-To":
            in_reply_to = values["value"]
    if in_reply_to is None:
        raise ValueError

    thread_id = msg["threadId"]

    thread = service.users().threads().get(userId="me", id=thread_id).execute()
    messages = thread["messages"]

    response_email = None
    for message in messages:
        email_data = message["payload"]["headers"]
        for values in email_data:
            if values["name"] == "Message-ID":
                message_id = values["value"]
                if message_id == in_reply_to:
                    response_email = message
    if response_email is None:
        raise ValueError
    starter_content = _extract_email_content(response_email)
    return ChatSession(messages=[starter_content, message_content])


class GMailLoader(BaseChatLoader):
    """Load data from `GMail`.

    There are many ways you could want to load data from GMail.
    This loader is currently fairly opinionated in how to do so.
    The way it does it is it first looks for all messages that you have sent.
    It then looks for messages where you are responding to a previous email.
    It then fetches that previous email, and creates a training example
    of that email, followed by your email.

    Note that there are clear limitations here. For example,
    all examples created are only looking at the previous email for context.

    To use:

    - Set up a Google Developer Account:
        Go to the Google Developer Console, create a project,
        and enable the Gmail API for that project.
        This will give you a credentials.json file that you'll need later.
    """

    def __init__(self, creds: Any, n: int = 100, raise_error: bool = False) -> None:
        super().__init__()
        self.creds = creds
        self.n = n
        self.raise_error = raise_error

    def lazy_load(self) -> Iterator[ChatSession]:
        from googleapiclient.discovery import build

        service = build("gmail", "v1", credentials=self.creds)
        results = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["SENT"], maxResults=self.n)
            .execute()
        )
        messages = results.get("messages", [])
        for message in messages:
            try:
                yield _get_message_data(service, message)
            except Exception as e:
                # TODO: handle errors better
                if self.raise_error:
                    raise e
                else:
                    pass
