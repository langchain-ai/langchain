from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import root_validator
from langchain.tools import BaseTool
from langchain.tools.ainetwork.app import AINAppOps
from langchain.tools.ainetwork.owner import AINOwnerOps
from langchain.tools.ainetwork.rule import AINRuleOps
from langchain.tools.ainetwork.transfer import AINTransfer
from langchain.tools.ainetwork.utils import authenticate
from langchain.tools.ainetwork.value import AINValueOps

if TYPE_CHECKING:
    from ain.ain import Ain


class AINetworkToolkit(BaseToolkit):
    """Toolkit for interacting with AINetwork Blockchain."""

    network: Optional[Literal["mainnet", "testnet"]] = "testnet"
    interface: Optional[Ain] = None

    @root_validator(pre=True)
    def set_interface(cls, values: dict) -> dict:
        if not values.get("interface"):
            values["interface"] = authenticate(network=values.get("network", "testnet"))
        return values

    class Config:
        """Pydantic config."""

        validate_all = True
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AINAppOps(),
            AINOwnerOps(),
            AINRuleOps(),
            AINTransfer(),
            AINValueOps(),
        ]
