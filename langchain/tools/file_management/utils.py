import sys
from pathlib import Path


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


def get_validated_relative_path(root: Path, user_path: str) -> Path:
    """Resolve a relative path, raising an error if not within the root directory."""
    # Note, this still permits symlinks from outside that point within the root.
    # Further validation would be needed if those are to be disallowed.
    root = root.resolve()
    full_path = (root / user_path).resolve()

    if not is_relative_to(full_path, root):
        raise ValueError(f"Path {user_path} is outside of the allowed directory {root}")
    return full_path
