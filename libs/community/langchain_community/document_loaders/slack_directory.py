import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class SlackDirectoryLoader(BaseLoader):
    """Load from a `Slack` directory dump."""

    def __init__(self, zip_path: str, workspace_url: Optional[str] = None):
        """Initialize the SlackDirectoryLoader.

        Args:
            zip_path (str): The path to the Slack directory dump zip file.
            workspace_url (Optional[str]): The Slack workspace URL.
              Including the URL will turn
              sources into links. Defaults to None.
        """
        self.zip_path = Path(zip_path)
        self.workspace_url = workspace_url
        self.channel_id_map = self._get_channel_id_map(self.zip_path)

    @staticmethod
    def _get_channel_id_map(zip_path: Path) -> Dict[str, str]:
        """Get a dictionary mapping channel names to their respective IDs."""
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            try:
                with zip_file.open("channels.json", "r") as f:
                    channels = json.load(f)
                return {channel["name"]: channel["id"] for channel in channels}
            except KeyError:
                return {}

    def lazy_load(self) -> Iterator[Document]:
        """Load and return documents from the Slack directory dump."""
        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            for channel_path in zip_file.namelist():
                channel_name = Path(channel_path).parent.name
                if not channel_name:
                    continue
                if channel_path.endswith(".json"):
                    messages = self._read_json(zip_file, channel_path)
                    for message in messages:
                        yield self._convert_message_to_document(message, channel_name)

    def _read_json(self, zip_file: zipfile.ZipFile, file_path: str) -> List[dict]:
        """Read JSON data from a zip subfile."""
        with zip_file.open(file_path, "r") as f:
            data = json.load(f)
        return data

    def _convert_message_to_document(
        self, message: dict, channel_name: str
    ) -> Document:
        """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
        text = message.get("text", "")
        metadata = self._get_message_metadata(message, channel_name)
        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _get_message_metadata(self, message: dict, channel_name: str) -> dict:
        """Create and return metadata for a given message and channel."""
        timestamp = message.get("ts", "")
        user = message.get("user", "")
        source = self._get_message_source(channel_name, user, timestamp)
        return {
            "source": source,
            "channel": channel_name,
            "timestamp": timestamp,
            "user": user,
        }

    def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
        """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
        if self.workspace_url:
            channel_id = self.channel_id_map.get(channel_name, "")
            return (
                f"{self.workspace_url}/archives/{channel_id}"
                + f"/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"
