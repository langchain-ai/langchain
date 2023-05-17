"""Loader that loads Obsidian directory dump."""
import re
import yaml
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class ObsidianLoader(BaseLoader):
    """Loader that loads Obsidian files from disk."""

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL)

    def __init__(
        self, path: str, encoding: str = "UTF-8", collect_metadata: bool = True
    ):
        """Initialize with path."""
        self.file_path = path
        self.encoding = encoding
        self.collect_metadata = collect_metadata

    def _parse_front_matter(self, content: str) -> dict:
        """Parse front matter metadata from the content and return it as a dict."""
        if not self.collect_metadata:
            return {}
        match = self.FRONT_MATTER_REGEX.search(content)
        front_matter = {}
        if match:
            front_matter = yaml.safe_load(match)
        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content."""
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def load(self) -> List[Document]:
        """Load documents."""
        obsidian_notes = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for note in obsidian_notes:
            with open(note, encoding=self.encoding) as file_stream:
                text = file_stream.read()

            front_matter = self._parse_front_matter(text)
            text = self._remove_front_matter(text)
            metadata = {
                "source": str(note.name),
                "path": str(note),
                "created": note.stat().st_ctime,
                "last_modified": note.stat().st_mtime,
                "last_accessed": note.stat().st_atime,
                **front_matter,
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
