import json
import logging
import os
from typing import Optional

import aiohttp
from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class DiscordClientWrapper:
    """HTTP client wrapper for Discord API interactions."""

    def __init__(self, token: str):
        self.token = token
        self._session: Optional[aiohttp.ClientSession] = None

    async def send_message(self, channel_id: int, message: str) -> str:
        """
        Sends a message to a Discord channel using the Discord REST API.

        Args:
            channel_id: The numeric ID of the target channel
            message: The message content to send

        Returns:
            Result message indicating success or failure
        """
        headers = {
            "Authorization": f"Bot {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"content": message, "allowed_mentions": {"parse": []}}
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        return "Message sent successfully!"
                    elif response.status == 403:
                        return "Error: Insufficient permissions to send message"
                    else:
                        error_text = await response.text()
                        return f"API Error ({response.status}): {error_text}"
        except aiohttp.ClientError as e:
            logger.error(f"Network error: {str(e)}")
            return f"Network error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Unexpected error: {str(e)}"

    async def read_messages(self, channel_id: int, limit: int = 50) -> str:
        """
        Reads messages from a Discord channel using the Discord REST API.

        Args:
            channel_id: The numeric ID of the target channel.
            limit: The number of messages to retrieve.

        Returns:
            A JSON-formatted string with the list of messages or an error message.
        """
        headers = {
            "Authorization": f"Bot {self.token}",
            "Content-Type": "application/json",
        }
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        params = {"limit": limit}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        messages = await response.json()
                        # Return the messages as a JSON-formatted string
                        return json.dumps(messages, ensure_ascii=False, indent=2)
                    elif response.status == 403:
                        return "Error: Insufficient permissions to read messages"
                    else:
                        error_text = await response.text()
                        return f"API Error ({response.status}): {error_text}"
        except aiohttp.ClientError as e:
            logger.error(f"Network error while reading messages: {str(e)}")
            return f"Network error while reading messages: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error while reading messages: {str(e)}")
            return f"Unexpected error while reading messages: {str(e)}"


def login() -> DiscordClientWrapper:
    """Factory function to create a Discord client wrapper."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
    return DiscordClientWrapper(token)


class DiscordBaseTool(BaseTool):  # type: ignore[override]
    """Base class for Discord tools."""

    client: DiscordClientWrapper = Field(default_factory=login)
    """The Discord client wrapper."""
