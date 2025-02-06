"""Base class for Slack tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.tools.slack.utils import login

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from slack_sdk import WebClient
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from slack_sdk import WebClient
    except ImportError:
        pass


class SlackBaseTool(BaseTool):  # type: ignore[override]
    """Base class for Slack tools."""

    client: WebClient = Field(default_factory=login)
    """The WebClient object."""
