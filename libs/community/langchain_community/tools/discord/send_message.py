import asyncio
import logging
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import DiscordBaseTool

logger = logging.getLogger(__name__)


class DiscordSendMessageSchema(BaseModel):
    """Input schema for Discord message sending tool."""

    message: str = Field(
        ...,
        description="Content of the message to send; supports basic Markdown "
        "formatting.",
    )
    channel_id: str = Field(
        ..., description="Numeric string representing the Discord channel ID."
    )


class DiscordSendMessage(DiscordBaseTool):
    """Tool for sending messages to Discord channels via the Discord API."""

    name: str = "discord_message_sender"
    description: str = (
        "Sends messages to specified Discord channels. "
        "Channel IDs must be numeric strings. "
        "Supports basic text formatting with Markdown."
    )
    args_schema: Type[BaseModel] | None = DiscordSendMessageSchema

    def _validate_inputs(self, message: str, channel_id: str) -> Optional[str]:
        """Common validation logic for both sync and async paths."""
        if not message.strip():
            return "Error: Message content cannot be empty"
        if not channel_id.isdigit():
            return "Error: Invalid channel ID format - must be numeric"
        return None

    def _run(
        self,
        message: str,
        channel_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution method."""
        validation_error = self._validate_inputs(message, channel_id)
        if validation_error:
            return validation_error
        try:
            result = asyncio.run(self.client.send_message(int(channel_id), message))
            return "Message result: " + result
        except Exception as e:
            logger.error(f"Error in sync send: {str(e)}")
            return f"Error sending message: {str(e)}"

    async def _arun(
        self,
        message: str,
        channel_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution method."""
        validation_error = self._validate_inputs(message, channel_id)
        if validation_error:
            return validation_error
        try:
            result = await self.client.send_message(int(channel_id), message)
            return "Message result: " + result
        except Exception as e:
            logger.error(f"Error in async send: {str(e)}")
            return f"Error sending message: {str(e)}"
