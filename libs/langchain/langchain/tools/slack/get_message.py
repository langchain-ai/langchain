import json
import logging
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.slack.base import SlackBaseTool


class SlackGetMessageSchema(BaseModel):
    """Input schema for SlackGetMessages."""

    channel_id: str = Field(
        ...,
        description="The channel id, private group, or IM channel to send message to.",
    )


class SlackGetMessage(SlackBaseTool):
    name: str = "get_messages"
    description: str = "Use this tool to get messages from a channel."

    args_schema: Type[SlackGetMessageSchema] = SlackGetMessageSchema

    def _run(
        self,
        channel_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logging.getLogger(__name__)
        try:
            result = self.client.conversations_history(channel=channel_id)
            messages = result["messages"]
            filtered_messages = [
                {key: message[key] for key in ("user", "text", "ts")}
                for message in messages
                if "user" in message and "text" in message and "ts" in message
            ]
            return json.dumps(filtered_messages)
        except Exception as e:
            return "Error creating conversation: {}".format(e)
