"""Base class for Gmail tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

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


class GmailBaseTool(BaseTool):
    api_resource: Resource = Field(default_factory=build_resource_service)

    @classmethod
    def from_api_resource(cls, api_resource: Resource) -> "GmailBaseTool":
        return cls(service=api_resource)
