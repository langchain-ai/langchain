import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def _read_json(json_file: Path) -> dict:
    """Read a json file"""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


class SlackDirectoryLoader(BaseLoader):
    """Loader for loading documents from a Slack directory dump."""

    def __init__(self, file_path: str, workspace_url: Optional[str] = None):
        """Initialize the SlackDirectoryLoader.

        Args:
            file_path (str): The path to the Slack directory dump.
            workspace_url (Optional[str]): The Slack workspace URL.
              Including the URL will turn
              sources into links. Defaults to None.
        """
        self.file_path = Path(file_path)
        self.workspace_url = workspace_url
        self.channel_id_map = self._get_channel_id_map()

    def _get_channel_id_map(self) -> Dict[str, str]:
        """Get a dictionary mapping channel names to their respective IDs."""
        channels_json_path = self.file_path / "channels.json"
        if channels_json_path.exists():
            with open(channels_json_path, encoding="utf-8") as f:
                channels = json.load(f)
            return {channel["name"]: channel["id"] for channel in channels}
        return {}

    def load(self) -> List[Document]:
        """Load and return documents from the Slack directory dump."""
        docs = []
        for channel_path in self.file_path.iterdir():
            if not channel_path.is_dir():
                continue
            channel_name = channel_path.name
            for json_file in channel_path.glob("*.json"):
                messages = _read_json(json_file)
                for message in messages:
                    document = self._convert_message_to_document(message, channel_name)
                    docs.append(document)
        return docs

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
                + "/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"
