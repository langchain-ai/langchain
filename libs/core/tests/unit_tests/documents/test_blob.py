from pathlib import Path

from langchain_core.documents import Blob


def test_blob_from_path_guesses_mimetype(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    blob = Blob.from_path(file_path)

    assert blob.mimetype == "text/plain"
    assert blob.path == file_path


def test_blob_from_path_skips_guessing(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"\x00\x01")

    blob = Blob.from_path(file_path, guess_type=False)

    assert blob.mimetype is None


def test_blob_from_path_prefers_explicit_mimetype(tmp_path: Path) -> None:
    file_path = tmp_path / "data.unknown"
    file_path.write_bytes(b"{}")

    blob = Blob.from_path(file_path, mime_type="application/custom", guess_type=True)

    assert blob.mimetype == "application/custom"


def test_blob_reads_string_from_path(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    blob = Blob.from_path(file_path)

    assert blob.as_string() == "hello"
