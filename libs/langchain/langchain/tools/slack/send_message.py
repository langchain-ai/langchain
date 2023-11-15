from typing import List, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool
from slack_sdk.errors import SlackApiError
import logging
class SendMessageSchema(BaseModel):
    """Input for SendMessageTool."""

    message: str = Field(
        ...,
        description="The message to be sent.",
    )
    channel: str = Field(
        ...,
        description="The channel, private group, or IM channel to send message to.",
    )


class SlackSendMessage(SlackBaseTool):
    """Tool for sending a message in Slack."""

    name: str = "send_message"
    description: str = (
        "Use this tool to send a message with the provided message fields."
    )
    args_schema: Type[SendMessageSchema] = SendMessageSchema

    def _run(
        self,
        message: str,
        channel: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger = logging.getLogger(__name__)
        logger.error("Message: %s", message)
        logger.error("Channel: %s", channel)
        result = self.client.chat_postMessage(channel=channel, text=message)
        output = "Message sent: " + str(result)
        # output = "msg sent"
        return output
        # try:
        #     result = self.client.chat_postMessage(channel=channel, text=message)
        #     output = "Message sent: " + str(result)
        #     return output
        # except SlackApiError as e:
        #     logger.error("Error: %s", e)
        #     logger.error("Failed to send message")
        #     return "Failed to send message"
        
