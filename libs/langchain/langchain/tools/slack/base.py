"""Base class for Slack tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.tools.slack.utils import login

if TYPE_CHECKING:
    from slack_sdk import WebClient


class SlackBaseTool(BaseTool):
    """Base class for Slack tools."""

    client: WebClient = Field(default_factory=login)
    """The WebClient object."""
