"""Base class for Gmail tools."""
from __future__ import annotations

from O365 import Account
from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.tools.office365.utils import authenticate


class O365BaseTool(BaseTool):
    account: Account = Field(default_factory=authenticate)
