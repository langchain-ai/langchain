from pathlib import Path

import pytest

from langchain.document_loaders import DirectoryLoader, TextLoader


@pytest.mark.requires("chardet")
def test_text_loader_detect_encodings() -> None:
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
