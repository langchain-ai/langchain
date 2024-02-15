from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from langchain_core.pydantic_v1 import root_validator

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.ainetwork.app import AINAppOps
from langchain_community.tools.ainetwork.owner import AINOwnerOps
from langchain_community.tools.ainetwork.rule import AINRuleOps
from langchain_community.tools.ainetwork.transfer import AINTransfer
from langchain_community.tools.ainetwork.utils import authenticate
from langchain_community.tools.ainetwork.value import AINValueOps

if TYPE_CHECKING:
    from ain.ain import Ain


class AINetworkToolkit(BaseToolkit):
    """Toolkit for interacting with AINetwork Blockchain.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

        See https://python.langchain.com/docs/security for more information.
    """

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
