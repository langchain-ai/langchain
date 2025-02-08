import asyncio
import logging
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from .base import DiscordClientWrapper, login

logger = logging.getLogger(__name__)


class DiscordReadMessagesSchema(BaseModel):
    """Input schema for Discord message reading tool."""

    channel_id: str = Field(
        ..., description="Numeric string representing the Discord channel ID."
    )
    limit: int = Field(
        50, description="Number of messages to retrieve (default is 50)."
    )


class DiscordReadMessages(BaseTool):
    """Tool for reading messages from a Discord channel via the Discord API."""

    name: str = "discord_message_reader"
    description: str = (
        "Reads messages from a specified Discord channel using the Discord REST API. "
        "Channel IDs must be numeric strings. Returns the messages as a JSON-formatted "
        "string."
    )
    args_schema: Type[DiscordReadMessagesSchema] = DiscordReadMessagesSchema
    _client: DiscordClientWrapper = PrivateAttr(default_factory=login)

    def _validate_inputs(self, channel_id: str, limit: int) -> Optional[str]:
        if not channel_id.isdigit():
            return "Error: Invalid channel ID format - must be numeric"
        if limit <= 0:
            return "Error: Limit must be a positive integer"
        return None

    def _run(
        self,
        channel_id: str,
        limit: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution method."""
        validation_error = self._validate_inputs(channel_id, limit)
        if validation_error:
            return validation_error
        try:
            result = asyncio.run(self._client.read_messages(int(channel_id), limit))
            return "Read messages result: " + result
        except Exception as e:
            logger.error(f"Error in discord read messages: {str(e)}")
            return f"Error reading messages: {str(e)}"

    async def _arun(
        self,
        channel_id: str,
        limit: int = 50,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution method."""
        validation_error = self._validate_inputs(channel_id, limit)
        if validation_error:
            return validation_error
        try:
            result = await self._client.read_messages(int(channel_id), limit)
            return "Read messages result: " + result
        except Exception as e:
            logger.error(f"Error in discord read messages: {str(e)}")
            return f"Error reading messages: {str(e)}"
