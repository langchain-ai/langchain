"""Base class for AINetwork tools."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langchain.tools.ainetwork.utils import authenticate
from langchain.tools.base import BaseTool
from pydantic import Field

if TYPE_CHECKING:
    from ain.ain import Ain


class AINBaseTool(BaseTool):
    """Base class for the AINetwork tools."""

    interface: Ain = Field(default_factory=authenticate)
    """The interface object for the AINetwork Blockchain."""

    def _run(self, *args, **kwargs):
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self._arun(*args, **kwargs))
        loop.close()
        return result
