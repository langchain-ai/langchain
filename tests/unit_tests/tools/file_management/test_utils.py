"""Test the File Management utils."""


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from langchain.tools.file_management.utils import get_validated_relative_path


def test_get_validated_relative_path_errs_on_absolute() -> None:
    """Safely resolve a path."""
    root = Path(__file__).parent
    user_path = "/bin/bash"
    matches = f"Path {user_path} is outside of the allowed directory {root}"
    with pytest.raises(ValueError, match=matches):
        get_validated_relative_path(root, user_path)


def test_get_validated_relative_path_errs_on_parent_dir() -> None:
    """Safely resolve a path."""
    root = Path(__file__).parent
    user_path = "data/sub/../../../sibling"
    matches = f"Path {user_path} is outside of the allowed directory {root}"
    with pytest.raises(ValueError, match=matches):
        get_validated_relative_path(root, user_path)


def test_get_validated_relative_path() -> None:
    """Safely resolve a path."""
    root = Path(__file__).parent
    user_path = "data/sub/file.txt"
    expected = root / user_path
    result = get_validated_relative_path(root, user_path)
    assert result == expected


def test_get_validated_relative_path_errs_for_symlink_outside_root() -> None:
    """Test that symlink pointing outside of root directory is not allowed."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        user_path = "symlink_outside_root"

        outside_path = Path("/bin/bash")
        symlink_path = root / user_path
        symlink_path.symlink_to(outside_path)

        matches = (
            f"Path {user_path} is outside of the allowed directory {root.resolve()}"
        )
        with pytest.raises(ValueError, match=matches):
            get_validated_relative_path(root, user_path)

        symlink_path.unlink()


def test_get_validated_relative_path_for_symlink_inside_root() -> None:
    """Test that symlink pointing inside the root directory is allowed."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        user_path = "symlink_inside_root"
        target_path = "data/sub/file.txt"

        symlink_path = root / user_path
        target_path_ = root / target_path
        symlink_path.symlink_to(target_path_)

        expected = target_path_.resolve()
        result = get_validated_relative_path(root, user_path)
        assert result == expected
        symlink_path.unlink()
