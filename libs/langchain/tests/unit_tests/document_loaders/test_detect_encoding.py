from pathlib import Path

import pytest

from langchain.document_loaders import CSVLoader, DirectoryLoader, TextLoader
from langchain.document_loaders.helpers import detect_file_encodings


@pytest.mark.requires("chardet")
def test_loader_detect_encoding_text() -> None:
    """Test text loader."""
    path = Path(__file__).parent.parent / "examples"
    files = path.glob("**/*.txt")
    loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader)
    loader_detect_encoding = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_kwargs={"autodetect_encoding": True},
        loader_cls=TextLoader,  # type: ignore
    )

    with pytest.raises((UnicodeDecodeError, RuntimeError)):
        loader.load()

    docs = loader_detect_encoding.load()
    assert len(docs) == len(list(files))


@pytest.mark.requires("chardet")
def test_loader_detect_encoding_csv() -> None:
    """Test csv loader."""
    path = Path(__file__).parent.parent / "examples"
    files = path.glob("**/*.csv")

    # Count the number of lines.
    row_count = 0
    for file in files:
        encodings = detect_file_encodings(str(file))
        for encoding in encodings:
            try:
                row_count += sum(1 for line in open(file, encoding=encoding.encoding))
                break
            except UnicodeDecodeError:
                continue
        # CSVLoader uses DictReader, and one line per file is a header,
        # so subtract the number of files.
        row_count -= 1

    loader = DirectoryLoader(
        str(path), glob="**/*.csv", loader_cls=CSVLoader  # type: ignore
    )
    loader_detect_encoding = DirectoryLoader(
        str(path),
        glob="**/*.csv",
        loader_kwargs={"autodetect_encoding": True},
        loader_cls=CSVLoader,  # type: ignore
    )

    with pytest.raises((UnicodeDecodeError, RuntimeError)):
        loader.load()

    docs = loader_detect_encoding.load()
    assert len(docs) == row_count


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
