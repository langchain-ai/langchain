import logging
import re
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ObsidianLoader(BaseLoader):
    """Load `Obsidian` files from directory."""

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
    TAG_REGEX = re.compile(r"[^\S\/]#([a-zA-Z_]+[-_/\w]*)")
    DATAVIEW_LINE_REGEX = re.compile(r"^\s*(\w+)::\s*(.*)$", re.MULTILINE)
    DATAVIEW_INLINE_BRACKET_REGEX = re.compile(r"\[(\w+)::\s*(.*)\]", re.MULTILINE)
    DATAVIEW_INLINE_PAREN_REGEX = re.compile(r"\((\w+)::\s*(.*)\)", re.MULTILINE)

    def __init__(
        self, path: str, encoding: str = "UTF-8", collect_metadata: bool = True
    ):
        """Initialize with a path.

        Args:
            path: Path to the directory containing the Obsidian files.
            encoding: Charset encoding, defaults to "UTF-8"
            collect_metadata: Whether to collect metadata from the front matter.
                Defaults to True.
        """
        self.file_path = path
        self.encoding = encoding
        self.collect_metadata = collect_metadata

    def _parse_front_matter(self, content: str) -> dict:
        """Parse front matter metadata from the content and return it as a dict."""
        if not self.collect_metadata:
            return {}

        match = self.FRONT_MATTER_REGEX.search(content)
        if not match:
            return {}

        try:
            front_matter = yaml.safe_load(match.group(1))

            # If tags are a string, split them into a list
            if "tags" in front_matter and isinstance(front_matter["tags"], str):
                front_matter["tags"] = front_matter["tags"].split(", ")

            return front_matter
        except yaml.parser.ParserError:
            logger.warning("Encountered non-yaml frontmatter")
            return {}

    def _to_langchain_compatible_metadata(self, metadata: dict) -> dict:
        """Convert a dictionary to a compatible with langchain."""
        result = {}
        for key, value in metadata.items():
            if type(value) in {str, int, float}:
                result[key] = value
            else:
                result[key] = str(value)
        return result

    def _parse_document_tags(self, content: str) -> set:
        """Return a set of all tags in within the document."""
        if not self.collect_metadata:
            return set()

        match = self.TAG_REGEX.findall(content)
        if not match:
            return set()

        return {tag for tag in match}

    def _parse_dataview_fields(self, content: str) -> dict:
        """Parse obsidian dataview plugin fields from the content and return it
        as a dict."""
        if not self.collect_metadata:
            return {}

        return {
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_LINE_REGEX.findall(content)
            },
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_INLINE_PAREN_REGEX.findall(content)
            },
            **{
                match[0]: match[1]
                for match in self.DATAVIEW_INLINE_BRACKET_REGEX.findall(content)
            },
        }

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content."""
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def load(self) -> List[Document]:
        """Load documents."""
        paths = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for path in paths:
            with open(path, encoding=self.encoding) as f:
                text = f.read()

            front_matter = self._parse_front_matter(text)
            tags = self._parse_document_tags(text)
            dataview_fields = self._parse_dataview_fields(text)
            text = self._remove_front_matter(text)
            metadata = {
                "source": str(path.name),
                "path": str(path),
                "created": path.stat().st_ctime,
                "last_modified": path.stat().st_mtime,
                "last_accessed": path.stat().st_atime,
                **self._to_langchain_compatible_metadata(front_matter),
                **dataview_fields,
            }

            if tags or front_matter.get("tags"):
                metadata["tags"] = ",".join(
                    tags | set(front_matter.get("tags", []) or [])
                )

            docs.append(Document(page_content=text, metadata=metadata))

        return docs
