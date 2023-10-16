"""Base class for Office 365 tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.tools.slack.utils import authenticate

if TYPE_CHECKING:
    # from O365 import Account
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError



class SlackBaseTool(BaseTool):
    """Base class for the Office 365 tools."""
    client: WebClient=Field(default_facotry=authenticate)
    # account: Account = Field(default_factory=authenticate)
    """The client object for slack_sdk WebClient"""
    """The account object for the Office 365 account."""
