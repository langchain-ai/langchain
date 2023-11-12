"""Slack tool utils."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # from O365 import Account
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    # from slack_bolt import App
    # from slack_bolt.adapter.socket_mode import SocketModeHandler
logger = logging.getLogger(__name__)


# def clean_body(body: str) -> str:
#     """Clean body of a message or event."""
#     try:
#         from bs4 import BeautifulSoup

#         try:
#             # Remove HTML
#             soup = BeautifulSoup(str(body), "html.parser")
#             body = soup.get_text()

#             # Remove return characters
#             body = "".join(body.splitlines())

#             # Remove extra spaces
#             body = " ".join(body.split())

#             return str(body)
#         except Exception:
#             return str(body)
#     except ImportError:
#         return str(body)


def authenticate() -> WebClient:
    """Authenticate using the Microsoft Grah API"""
    try:
        logger.error("slack auth start")
        from slack_sdk import WebClient
        client = WebClient(token=os.environ.get("SLACK_USR_TOKEN"))
        logger.error("get client")
        return client
    except ImportError as e:
        # raise ImportError(
        #     "Cannot import slack_sdk. Please install the package with `pip install slack_sdk`."
        # ) from e
        logger.error("slack auth failed")
    
    
    # app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
    # SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()



  



    # if "CLIENT_ID" in os.environ and "CLIENT_SECRET" in os.environ:
    #     client_id = os.environ["CLIENT_ID"]
    #     client_secret = os.environ["CLIENT_SECRET"]
    #     credentials = (client_id, client_secret)
    # else:
    #     logger.error(
    #         "Error: The CLIENT_ID and CLIENT_SECRET environmental variables have not "
    #         "been set. Visit the following link on how to acquire these authorization "
    #         "tokens: https://learn.microsoft.com/en-us/graph/auth/"
    #     )
    #     return None

    # account = Account(credentials)

    # if account.is_authenticated is False:
    #     if not account.authenticate(
    #         scopes=[
    #             "https://graph.microsoft.com/Mail.ReadWrite",
    #             "https://graph.microsoft.com/Mail.Send",
    #             "https://graph.microsoft.com/Calendars.ReadWrite",
    #             "https://graph.microsoft.com/MailboxSettings.ReadWrite",
    #         ]
    #     ):
    #         print("Error: Could not authenticate")
    #         return None
    #     else:
    #         return account
    # else:
    #     return account


# def authenticate() -> Account:
#     """Authenticate using the Microsoft Grah API"""
#     try:
#         from O365 import Account
#     except ImportError as e:
#         raise ImportError(
#             "Cannot import 0365. Please install the package with `pip install O365`."
#         ) from e

#     if "CLIENT_ID" in os.environ and "CLIENT_SECRET" in os.environ:
#         client_id = os.environ["CLIENT_ID"]
#         client_secret = os.environ["CLIENT_SECRET"]
#         credentials = (client_id, client_secret)
#     else:
#         logger.error(
#             "Error: The CLIENT_ID and CLIENT_SECRET environmental variables have not "
#             "been set. Visit the following link on how to acquire these authorization "
#             "tokens: https://learn.microsoft.com/en-us/graph/auth/"
#         )
#         return None

#     account = Account(credentials)

#     if account.is_authenticated is False:
#         if not account.authenticate(
#             scopes=[
#                 "https://graph.microsoft.com/Mail.ReadWrite",
#                 "https://graph.microsoft.com/Mail.Send",
#                 "https://graph.microsoft.com/Calendars.ReadWrite",
#                 "https://graph.microsoft.com/MailboxSettings.ReadWrite",
#             ]
#         ):
#             print("Error: Could not authenticate")
#             return None
#         else:
#             return account
#     else:
#         return account
