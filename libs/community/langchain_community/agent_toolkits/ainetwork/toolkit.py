from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, model_validator

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

    Parameters:
        network: Optional. The network to connect to. Default is "testnet".
            Options are "mainnet" or "testnet".
        interface: Optional. The interface to use. If not provided, will
            attempt to authenticate with the network. Default is None.
    """

    network: Optional[Literal["mainnet", "testnet"]] = "testnet"
    interface: Optional[Ain] = None

    @model_validator(mode="before")
    @classmethod
    def set_interface(cls, values: dict) -> Any:
        """Set the interface if not provided.

        If the interface is not provided, attempt to authenticate with the
        network using the network value provided.

        Args:
            values: The values to validate.

        Returns:
            The validated values.
        """
        if not values.get("interface"):
            values["interface"] = authenticate(network=values.get("network", "testnet"))
        return values

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AINAppOps(),
            AINOwnerOps(),
            AINRuleOps(),
            AINTransfer(),
            AINValueOps(),
        ]
