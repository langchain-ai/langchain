"""Send Slack messages."""

from typing import List, Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool

class SendMessageSchema(BaseModel):
    """Input for SendMessageTool."""

    body: str = Field(
        ...,
        description="The formatted text of the message to be published.",
    )
    to: str = Field(
        ...,
        description="Channel, private group, or IM channel to send message to.",
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
            body: str,
            to: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        result = self.client.chat_postMessage(
            channel=to, 
            text=body
        )

        output = "Message sent: " + str(result)
        return output

