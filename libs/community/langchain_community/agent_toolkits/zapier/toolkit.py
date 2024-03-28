"""[DEPRECATED] Zapier Toolkit."""
from typing import List

from langchain_core._api import warn_deprecated
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.zapier.tool import ZapierNLARunAction
from langchain_community.utilities.zapier import ZapierNLAWrapper


class ZapierToolkit(BaseToolkit):
    """Zapier Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_zapier_nla_wrapper(
        cls, zapier_nla_wrapper: ZapierNLAWrapper
    ) -> "ZapierToolkit":
        """Create a toolkit from a ZapierNLAWrapper."""
        actions = zapier_nla_wrapper.list()
        tools = [
            ZapierNLARunAction(
                action_id=action["id"],
                zapier_description=action["description"],
                params_schema=action["params"],
                api_wrapper=zapier_nla_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    @classmethod
    async def async_from_zapier_nla_wrapper(
        cls, zapier_nla_wrapper: ZapierNLAWrapper
    ) -> "ZapierToolkit":
        """Create a toolkit from a ZapierNLAWrapper."""
        actions = await zapier_nla_wrapper.alist()
        tools = [
            ZapierNLARunAction(
                action_id=action["id"],
                zapier_description=action["description"],
                params_schema=action["params"],
                api_wrapper=zapier_nla_wrapper,
            )
            for action in actions
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        warn_deprecated(
            since="0.0.319",
            message=(
                "This tool will be deprecated on 2023-11-17. See "
                "https://nla.zapier.com/sunset/ for details"
            ),
        )
        return self.tools
