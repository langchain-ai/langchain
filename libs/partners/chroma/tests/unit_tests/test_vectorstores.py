import os
from pathlib import Path

import chromadb
import pytest
from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


def _make_store(*, allowed_image_dir: Path | str | None = None) -> Chroma:
    return Chroma(
        client=chromadb.Client(),
        embedding_function=None,
        allowed_image_dir=allowed_image_dir,
    )


class TestValidateImageUri:
    def test_path_traversal_blocked(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError, match="traversal"):
            store._validate_image_uri("../../../../etc/passwd")

    def test_absolute_path_blocked(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError, match="Absolute"):
            store._validate_image_uri("/etc/passwd")

    def test_relative_safe_path_allowed(self) -> None:
        store = _make_store()
        # plain relative path without .. must not raise
        store._validate_image_uri("images/photo.jpg")

    def test_valid_image_path_with_allowed_dir(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
        store = _make_store(allowed_image_dir=tmp_path)
        store._validate_image_uri(str(img))

    def test_path_outside_allowed_dir_blocked(self, tmp_path: Path) -> None:
        allowed = tmp_path / "images"
        allowed.mkdir()
        store = _make_store(allowed_image_dir=allowed)
        with pytest.raises(ValueError, match="outside the allowed directory"):
            store._validate_image_uri(str(tmp_path / "secret.txt"))

    def test_dotdot_escape_from_allowed_dir_blocked(self, tmp_path: Path) -> None:
        allowed = tmp_path / "images"
        allowed.mkdir()
        store = _make_store(allowed_image_dir=allowed)
        with pytest.raises(ValueError, match="outside the allowed directory"):
            store._validate_image_uri(str(allowed / ".." / "secret.txt"))

    def test_symlink_bypass_blocked(self, tmp_path: Path) -> None:
        allowed = tmp_path / "images"
        allowed.mkdir()
        target = tmp_path / "secret.txt"
        target.write_text("sensitive")
        link = allowed / "link.jpg"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        store = _make_store(allowed_image_dir=allowed)
        with pytest.raises(ValueError, match="outside the allowed directory"):
            store._validate_image_uri(str(link))


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1
