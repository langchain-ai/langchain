import sys
from pathlib import Path
from typing import Optional

from langchain.pydantic_v1 import BaseModel


def is_relative_to(path: Path, root: Path) -> bool:
    """Check if path is relative to root."""
    if sys.version_info >= (3, 9):
        # No need for a try/except block in Python 3.8+.
        return path.is_relative_to(root)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


INVALID_PATH_TEMPLATE = (
    "Error: Access denied to {arg_name}: {value}."
    " Permission granted exclusively to the current working directory"
)


class FileValidationError(ValueError):
    """Error for paths outside the root directory."""


class BaseFileToolMixin(BaseModel):
    """Mixin for file system tools."""

    root_dir: Optional[str] = None
    """The final path will be chosen relative to root_dir if specified."""

    def get_relative_path(self, file_path: str) -> Path:
        """Get the relative path, returning an error if unsupported."""
        if self.root_dir is None:
            return Path(file_path)
        return get_validated_relative_path(Path(self.root_dir), file_path)


def get_validated_relative_path(root: Path, user_path: str) -> Path:
    """Resolve a relative path, raising an error if not within the root directory."""
    # Note, this still permits symlinks from outside that point within the root.
    # Further validation would be needed if those are to be disallowed.
    root = root.resolve()
    full_path = (root / user_path).resolve()

    if not is_relative_to(full_path, root):
        raise FileValidationError(
            f"Path {user_path} is outside of the allowed directory {root}"
        )
    return full_path
