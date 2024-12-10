"""Base class for Google Calendar tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.tools.google_calendar.utils import build_resource_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


class GoogleCalendarBaseTool(BaseTool):  # type: ignore[override]
    """Base class for Google Calendar tools."""

    api_resource: Resource = Field(default_factory=build_resource_service)

    @classmethod
    def from_api_resource(cls, api_resource: Resource) -> "GoogleCalendarBaseTool":
        """Create a tool from an api resource.

        Args:
            api_resource: The api resource to use.

        Returns:
            A tool.
        """
        return cls(service=api_resource)  # type: ignore[call-arg]
