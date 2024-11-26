from pathlib import Path
from typing import Any, Iterator, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.directory import DirectoryLoader


def _is_visible(p: Path) -> bool:
    """Check if a path is visible (not hidden)."""
    parts = p.parts
    for _p in parts:
        if _p.startswith("."):
            return False
    return True


class URLContentLoader(DirectoryLoader):
    """Specialized loader to handle URLs with line breaks and special characters."""

    def read_file(self, file_path: Path) -> str:
        """Read the content of the file and return it as a string."""
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()

    def _fix_special_chars_in_urls(self, content: str) -> str:
        """
        Fix line breaks in URLs with special characters directly within content.
        Ensures each URL is printed on a separate line.
        """
        lines = content.splitlines()
        processed_lines = []
        current_url = ""

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("http://") or stripped_line.startswith(
                "https://"
            ):
                if current_url:
                    processed_lines.append(current_url)
                current_url = stripped_line
            elif current_url:
                current_url += stripped_line
            else:
                processed_lines.append(stripped_line)

        if current_url:
            processed_lines.append(current_url)
        return "\n".join(processed_lines)

    def _lazy_load_file(
        self, item: Path, path: Path, pbar: Optional[Any]
    ) -> Iterator[Document]:
        """Load a file with specialized URL handling."""
        if item.is_file():
            if _is_visible(item.relative_to(path)) or self.load_hidden:
                try:
                    content = self.read_file(item)
                    processed_content = self._fix_special_chars_in_urls(content)
                    for line in processed_content.splitlines():
                        yield Document(
                            page_content=line, metadata={"source": str(item)}
                        )
                except Exception as e:
                    if self.silent_errors:
                        pass
                    else:
                        raise e
