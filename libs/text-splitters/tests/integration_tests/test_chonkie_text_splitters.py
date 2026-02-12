"""Test ChonkieTextSplitter functionality."""

import pytest
from chonkie.pipeline import ComponentRegistry  # type: ignore[import]

from langchain_text_splitters.chonkie import ChonkieTextSplitter

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("chonkie", reason="chonkie not installed"),
    reason="chonkie not installed",
)


def test_chonkie_text_splitter_basic() -> None:
    """Test basic splitting with default chunker (recursive)."""
    text = "This is a test. Here is another sentence."
    splitter = ChonkieTextSplitter()
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert any("test" in chunk for chunk in chunks)


def test_chonkie_text_splitter_with_alias() -> None:
    """Test splitting with a specific valid chunker alias."""
    # Use the first available alias for testing
    aliases = ChonkieTextSplitter.valid_chunker_aliases
    assert "recursive" in aliases
    text = "Sentence one. Sentence two. Sentence three."
    splitter = ChonkieTextSplitter(chunker="recursive")
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chonkie_text_splitter_with_chunker_instance() -> None:
    """Test passing a chunker instance directly."""
    ChunkingClass = ComponentRegistry.get_chunker("recursive").component_class  # noqa: N806
    chunker_instance = ChunkingClass()
    text = "Direct instance test."
    splitter = ChonkieTextSplitter(chunker=chunker_instance)
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chonkie_text_splitter_invalid_alias() -> None:
    """Test that an invalid chunker alias raises an error."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        ChonkieTextSplitter(chunker="not_a_real_chunker")


def test_chonkie_text_splitter_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ImportError is raised if chonkie is not installed."""
    import sys  # noqa: PLC0415

    # Simulate chonkie not installed
    monkeypatch.setitem(sys.modules, "chonkie", None)
    monkeypatch.setitem(sys.modules, "chonkie.chunker.base", None)
    monkeypatch.setitem(sys.modules, "chonkie.pipeline", None)
    # Force reload
    from importlib import reload  # noqa: PLC0415

    import langchain_text_splitters.chonkie as chonkie_mod  # noqa: PLC0415

    reload(chonkie_mod)
    with pytest.raises(ImportError):
        chonkie_mod.ChonkieTextSplitter()
