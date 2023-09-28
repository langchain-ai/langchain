from pathlib import Path

import pytest

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.helpers import detect_file_encodings


@pytest.mark.requires("chardet")
def test_loader_detect_encoding() -> None:
    """Test text loader."""
    path = Path(__file__).parent.parent / "examples"
    files = path.glob("**/*.txt")
    loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader)
    loader_detect_encoding = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_kwargs={"autodetect_encoding": True},
        loader_cls=TextLoader,
    )

    with pytest.raises((UnicodeDecodeError, RuntimeError)):
        loader.load()

    docs = loader_detect_encoding.load()
    assert len(docs) == len(list(files))


@pytest.mark.skip(reason="slow test")
@pytest.mark.requires("chardet")
def test_loader_detect_encoding_timeout(tmpdir: str) -> None:
    path = Path(tmpdir)
    file_path = str(path / "blob.txt")
    # 2mb binary blob
    with open(file_path, "wb") as f:
        f.write(b"\x00" * 2_000_000)

    with pytest.raises(TimeoutError):
        detect_file_encodings(file_path, timeout=1)

    detect_file_encodings(file_path, timeout=10)
