from __future__ import annotations
from typing import List, Any
import os
import base64
import re
import json
from bs4 import BeautifulSoup

import email
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from langchain.agents.tools import Tool
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.agents.agent_toolkits.gmail.prompt import (
    QUERY_PROMPT,
    DRAFT_PROMPT,
    SEND_PROMPT,
)
from langchain.prompts import PromptTemplate

from datetime import datetime


SCOPES = ["https://mail.google.com/"]


class GmailToolkit(BaseToolkit):
    """Toolkit for interacting with Gmail."""

    credentials: dict = None
    llm: BaseLLM = None
    sender_name: str = None
    credentials: dict = None
    service: Any = None
    tools_return_direct: bool = False

    # initialize the toolkit
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.credentials is None:
            self.credentials = self._get_credentials()

        self.service = build("gmail", "v1", credentials=self.credentials)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        search_messages = Tool(
            name="search_messages",
            description="Use this tool to search for email messages. The input must be a valid Gmail query. The output is a JSON list of messages.",
            func=self.search_messages,
            return_direct=self.tools_return_direct,
        )

        search_threads = Tool(
            name="search_threads",
            description="Use this tool to search for email threads. The input must be a valid Gmail query. The output is a JSON list of messages.",
            func=self.search_threads,
            return_direct=self.tools_return_direct,
        )

        create_draft = Tool(
            name="create_draft",
            description="Use this tool to create a draft email. ",
            func=self.create_draft,
            return_direct=self.tools_return_direct,
        )

        send_message = Tool(
            name="send_message",
            description="Use this tool to send an email",
            func=self.send_message,
            return_direct=self.tools_return_direct,
        )

        get_message = Tool(
            name="get_message",
            description="Use this tool to get an email message. The input must be the unique ID of a message. The output is a JSON message object.",
            func=self.get_message,
            return_direct=self.tools_return_direct,
        )

        get_thread = Tool(
            name="get_thread",
            description="Use this tool to get an email thread. The input must be the unique ID of a thread. The output is a JSON thread object",
            func=self.get_thread,
            return_direct=self.tools_return_direct,
        )

        return [
            search_messages,
            search_threads,
            create_draft,
            send_message,
            get_message,
            get_thread,
        ]

    def _get_datetime(self):
        now = datetime.now()
        return now.strftime("%m/%d/%Y, %H:%M:%S")

    def _get_credentials(self) -> dict:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def create_draft(self, input):
        prompt = DRAFT_PROMPT.format(
            input=input, sender_name=self.sender_name, date=self._get_datetime()
        )

        try:
            email_text = self.llm(prompt)
            email_regex = r"To: (\S+)"
            subject_regex = r"Subject: (.+)"
            message_regex = r"Message:([\s\S]+)"

            recipient = re.search(email_regex, email_text).group(1).strip()
            subject = re.search(subject_regex, email_text).group(1).strip()
            body = re.search(message_regex, email_text).group(1).strip()

            if not recipient or not subject or not body:
                raise ValueError(
                    "Invalid email format. Recipient, subject, and body are required."
                )

            message = EmailMessage()

            message.set_content(body)

            message["To"] = recipient
            message["Subject"] = subject

            # encoded message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            create_message = {"message": {"raw": encoded_message}}

            draft = (
                self.service.users()
                .drafts()
                .create(userId="me", body=create_message)
                .execute()
            )

            output = f"Draft created. {draft['id']}\nDraft message: {draft['message']}"
            return output
        except Exception as e:
            raise Exception(f"An error occurred: {e}")

    def send_message(self, input):
        """Create and send an email message
        Print the returned  message id
        Returns: Message object, including message id
        """

        prompt = SEND_PROMPT.format(
            input=input, sender_name=self.sender_name, date=self._get_datetime()
        )

        try:
            email_text = self.llm(prompt)

            email_regex = r"To: (\S+)"
            subject_regex = r"Subject: (.+)"
            message_regex = r"Message:([\s\S]+)"

            recipient = re.search(email_regex, email_text).group(1).strip()
            subject = re.search(subject_regex, email_text).group(1).strip()
            body = re.search(message_regex, email_text).group(1).strip()

            if not recipient or not subject or not body:
                raise ValueError(
                    "Invalid email format. Recipient, subject, and body are required."
                )

            message = MIMEMultipart()
            message.attach(MIMEText(body, "html"))

            message["To"] = recipient
            message["Subject"] = subject

            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            create_message = {"raw": encoded_message}

            send_message = (
                self.service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute()
            )
            output = f'Message sent. Message Id: {send_message["id"]}'
            return output
        except Exception as error:
            raise Exception(f"An error occurred: {error}")

    def search_all(self, input):
        if "threads" in input:
            return self.search_threads(input)
        elif "emails" in input:
            return self.search_messages(input)
        else:
            return self.search_threads(input)

    def search_threads(self, input):
        prompt = QUERY_PROMPT.format(input=input, date=self._get_datetime())

        query = self.llm(prompt)

        threads = (
            self.service.users()
            .threads()
            .list(userId="me", q=query, maxResults=10)
            .execute()
            .get("threads", [])
        )

        results = []
        # Print the thread IDs and snippets
        for thread in threads:
            thread_id = thread["id"]
            thread_data = (
                self.service.users().threads().get(userId="me", id=thread_id).execute()
            )
            messages = thread_data["messages"]
            thread["messages"] = []
            for message in messages:
                snippet = message["snippet"]
                thread["messages"].append({"snippet": snippet, "id": message["id"]})
            results.append(thread)

        return json.dumps(results)

    def search_messages(self, input):
        prompt = QUERY_PROMPT.format(input=input, date=self._get_datetime())

        query = self.llm(prompt)

        print("Query: ", query)

        service = build("gmail", "v1", credentials=self.credentials)

        messages = (
            service.users()
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

            body = self.clean_email_body(message_body)

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

    def get_message(self, message_id):
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

        body = self.clean_email_body(message_body)

        return json.dumps(
            {
                "id": message_id,
                "threadId": message_data["threadId"],
                "snippet": message_data["snippet"],
                "body": body,
                "subject": subject,
                "sender": sender,
            }
        )

    def get_thread(self, thread_id):
        thread_data = (
            self.service.users().threads().get(userId="me", id=thread_id).execute()
        )

        return json.dumps(thread_data)

    def clean_email_body(self, body: str):
        try:
            soup = BeautifulSoup(str(body), "html.parser")
            body = soup.get_text()
            return str(body)
        except Exception as e:
            raise Exception("Can't parse message body" + str(e))
