import json
from pathlib import Path
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class SlackDirectoryLoader(BaseLoader):
    """Loader that loads documents from Slack directory dump."""

    def __init__(self, path: str, workspace_url: Optional[str] = None):
        """Initialize with path and optional workspace URL.

        Including the URL will turn sources into links
        """
        self.file_path = path
        self.workspace_url = workspace_url
        self.channel_id_map = self._get_channel_id_map()

    def _get_channel_id_map(self) -> Dict[str, str]:
        """Get a dictionary mapping channel names to their respective IDs."""
        channels_json_path = Path(self.file_path) / "channels.json"
        if channels_json_path.exists():
            with open(channels_json_path, encoding="utf-8") as f:
                channels = json.load(f)
            return {channel["name"]: channel["id"] for channel in channels}
        return {}

    def load(self) -> List[Document]:
        """Load documents."""
        channel_paths = list(Path(self.file_path).glob("*"))
        docs = []
        for channel_path in channel_paths:
            if channel_path.is_dir():
                channel_name = channel_path.name
                json_files = list(channel_path.glob("*.json"))
                for json_file in json_files:
                    with open(json_file, encoding="utf-8") as f:
                        messages = json.load(f)
                    for message in messages:
                        text = message.get("text", "")
                        timestamp = message.get("ts")
                        user = message.get("user")
                        if self.workspace_url:
                            channel_id = self.channel_id_map.get(channel_name, "")
                            message_link = (
                                f"{self.workspace_url}/archives/"
                                + f"{channel_id}/p{timestamp.replace('.', '')}"
                            )
                            source = message_link
                        else:
                            source = f"{channel_name} - {user} - {timestamp}"
                        metadata = {
                            "source": source,
                            "channel": channel_name,
                            "timestamp": timestamp,
                            "user": user,
                        }
                        docs.append(Document(page_content=text, metadata=metadata))
        return docs
