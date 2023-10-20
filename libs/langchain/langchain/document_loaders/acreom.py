import re
from pathlib import Path
from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class AcreomLoader(BaseLoader):
    """Load `acreom` vault from a directory."""

    FRONT_MATTER_REGEX = re.compile(r"^---\n(.*?)\n---\n", re.MULTILINE | re.DOTALL)
    """Regex to match front matter metadata in markdown files."""

    def __init__(
        self, path: str, encoding: str = "UTF-8", collect_metadata: bool = True
    ):
        """Initialize the loader."""
        self.file_path = path
        """Path to the directory containing the markdown files."""
        self.encoding = encoding
        """Encoding to use when reading the files."""
        self.collect_metadata = collect_metadata
        """Whether to collect metadata from the front matter."""

    def _parse_front_matter(self, content: str) -> dict:
        """Parse front matter metadata from the content and return it as a dict."""
        if not self.collect_metadata:
            return {}
        match = self.FRONT_MATTER_REGEX.search(content)
        front_matter = {}
        if match:
            lines = match.group(1).split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    front_matter[key.strip()] = value.strip()
                else:
                    # Skip lines without a colon
                    continue
        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter metadata from the given content."""
        if not self.collect_metadata:
            return content
        return self.FRONT_MATTER_REGEX.sub("", content)

    def _process_acreom_content(self, content: str) -> str:
        # remove acreom specific elements from content that
        # do not contribute to the context of current document
        content = re.sub(r"\s*-\s\[\s\]\s.*|\s*\[\s\]\s.*", "", content)  # rm tasks
        content = re.sub(r"#", "", content)  # rm hashtags
        content = re.sub(r"\[\[.*?\]\]", "", content)  # rm doclinks
        return content

    def lazy_load(self) -> Iterator[Document]:
        ps = list(Path(self.file_path).glob("**/*.md"))

        for p in ps:
            with open(p, encoding=self.encoding) as f:
                text = f.read()

            front_matter = self._parse_front_matter(text)
            text = self._remove_front_matter(text)

            text = self._process_acreom_content(text)

            metadata = {
                "source": str(p.name),
                "path": str(p),
                **front_matter,
            }

            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        return list(self.lazy_load())
