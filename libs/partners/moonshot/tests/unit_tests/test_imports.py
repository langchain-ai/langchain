"""Test that imports work correctly."""

from __future__ import annotations


def test_import_core() -> None:
    """Test that the main module imports correctly."""
    from langchain_moonshot import ChatMoonshot, __version__  # noqa: F811

    assert __version__ == "0.1.0"
    assert ChatMoonshot is not None


def test_import_chat_models() -> None:
    """Test that chat_models submodule imports correctly."""
    from langchain_moonshot.chat_models import ChatMoonshot  # noqa: F811

    assert ChatMoonshot.__name__ == "ChatMoonshot"


def test_import_data() -> None:
    """Test that data submodule imports correctly."""
    from langchain_moonshot.data import _PROFILES  # noqa: F811

    assert "moonshot-v1-8k" in _PROFILES
    assert "moonshot-v1-128k" in _PROFILES
