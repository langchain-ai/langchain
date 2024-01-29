"""Tool for the Google Trends"""
import json

from typing import Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.infobip import InfobipAPIWrapper


class InfobipTool(BaseTool):
    """Tool that sends SMS or Email message via Infobip."""

    name: str = "infobip"
    description: str = (
        "A wrapper around Infobip API. "
        "This tool is used to send SMS or Email messages"
        "If you are asked to send a message, you can use this tool to send a message to a phone number or email address"
        "For example, to send an SMS message with text 'test message' to number +1234567890, you would pass in the following dictionary: {\"message\": \"test message\", \"to\": \"+1234567890\"}"
        "For example, to send an Email message with text 'test message' to email address to email example@example.com with subject 'test subject', you would pass in the following dictionary: {\"message\": \"test message\", \"to\": \"example@example.com\", \"subject\": \"test subject\"}"
    )
    api_wrapper: InfobipAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use Infobip API to send SMS or Email message.""" 
        data: Dict = json.loads(query)
        return self.api_wrapper.run(
            message=data["message"],
            to=data["to"],
            subject=data.get("subject", "")
        )
