"""Base class for Gmail tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from langchain.tools.base import BaseTool
from langchain.tools.office365.utils import authenticate

if TYPE_CHECKING:
    from O365 import Account


class O365BaseTool(BaseTool):
    account: Account = Field(default_factory=authenticate)
