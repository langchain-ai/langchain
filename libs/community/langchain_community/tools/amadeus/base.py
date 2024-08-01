"""Base class for Amadeus tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.tools.amadeus.utils import authenticate

if TYPE_CHECKING:
    from amadeus import Client


class AmadeusBaseTool(BaseTool):
    """Base Tool for Amadeus."""

    client: Client = Field(default_factory=authenticate)
