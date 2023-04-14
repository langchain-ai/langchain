"""Code to load blobs from a file system."""
from pathlib import Path
from typing import Generator

from langchain.document_loaders.base import Blob, BlobLoader


def _is_visible(p: Path) -> bool:
    """Is the given path visible."""
    parts = p.parts
    for _p in parts:
        if _p.startswith("."):
            return False
    return True


class FileSystemLoader(BlobLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(
        self,
        path: str,
        glob: str = "**/[!.]*",
        *,
        load_hidden: bool = False,
        recursive: bool = False,
    ):
        """Initialize with path to directory and how to glob over it."""
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive

    def yield_blobs(
        self,
    ) -> Generator[Blob, None, None]:
        """Yield blobs that match the requested pattern."""
        p = Path(self.path)
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for item in items:
            if item.is_file():
                if _is_visible(item.relative_to(p)) or self.load_hidden:
                    yield Blob.from_path(str(item))
