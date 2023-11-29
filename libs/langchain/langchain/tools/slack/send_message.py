from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool


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
        try:
            result = self.client.chat_postMessage(channel=channel, text=message)
            output = "Message sent: " + str(result)
            return output
        except Exception as e:
            return "Error creating conversation: {}".format(e)
