from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.ainetwork.get_value import AINGetValue
from langchain.tools.ainetwork.set_function import AINSetFunction
from langchain.tools.ainetwork.set_value import AINSetValue
from langchain.tools.ainetwork.transfer import AINTransfer
from langchain.tools.ainetwork.utils import authenticate
from pydantic import Field

if TYPE_CHECKING:
    from ain.ain import Ain


class AINetworkToolkit(BaseToolkit):
    """Toolkit for interacting with AINetwork Blockchain."""

    interface: Ain = Field(default_factory=authenticate)

    class Config:
        """Pydantic config."""

        validate_all = True
        arbitrary_types_allowed = False

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AINSetValue(interface=self.interface),
            AINGetValue(interface=self.interface),
            AINSetFunction(interface=self.interface),
            AINTransfer(interface=self.interface),
        ]
