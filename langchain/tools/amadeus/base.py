"""Base class for Amadeus tools."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from langchain.tools.amadeus.utils import authenticate
from langchain.tools.base import BaseTool

if TYPE_CHECKING:
    from amadeus import Client


class AmadeusBaseTool(BaseTool):
    client: Client = Field(default_factory=authenticate)
