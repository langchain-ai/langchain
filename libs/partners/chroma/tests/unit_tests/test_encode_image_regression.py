"""Regression tests for path traversal in Chroma.encode_image() (#37296).

These tests verify that encode_image rejects path traversal sequences
and non-image file extensions.
"""

import tempfile
from pathlib import Path

import pytest

from langchain_chroma.vectorstores import Chroma


class TestEncodeImagePathTraversal:
    """Regression tests for #37296 — path traversal via unsanitized URI."""

    def test_path_traversal_with_double_dots_blocked(self) -> None:
        """URIs containing '..' must be rejected."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            Chroma.encode_image("../../../../etc/passwd")

    def test_path_traversal_relative_blocked(self) -> None:
        """Relative paths with '..' must be rejected."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            Chroma.encode_image("../../../secret.png")

    def test_path_traversal_mixed_separators_blocked(self) -> None:
        """Mixed path separators with '..' must be rejected."""
        with pytest.raises(ValueError, match="Path traversal detected"):
            Chroma.encode_image("images/../../etc/shadow")

    def test_non_image_extension_blocked(self) -> None:
        """Non-image file extensions must be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported image file extension"):
                Chroma.encode_image(f.name)

    def test_no_extension_blocked(self) -> None:
        """Files without extensions must be rejected."""
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported image file extension"):
                Chroma.encode_image(f.name)

    def test_valid_png_accepted(self) -> None:
        """A legitimate .png file should be accepted."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG header
            f.flush()
            result = Chroma.encode_image(f.name)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_valid_jpg_accepted(self) -> None:
        """A legitimate .jpg file should be accepted."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")  # JPEG header
            f.flush()
            result = Chroma.encode_image(f.name)
            assert isinstance(result, str)

    def test_valid_webp_accepted(self) -> None:
        """A legitimate .webp file should be accepted."""
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WEBP")
            f.flush()
            result = Chroma.encode_image(f.name)
            assert isinstance(result, str)

    def test_case_insensitive_extension(self) -> None:
        """Extension check should be case-insensitive."""
        with tempfile.NamedTemporaryFile(suffix=".PNG", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.flush()
            result = Chroma.encode_image(f.name)
            assert isinstance(result, str)

    def test_passwd_file_blocked(self) -> None:
        """Explicit /etc/passwd path must be rejected."""
        with pytest.raises(ValueError):
            Chroma.encode_image("/etc/passwd")

    def test_python_source_blocked(self) -> None:
        """Python source files must be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"import os")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported image file extension"):
                Chroma.encode_image(f.name)
