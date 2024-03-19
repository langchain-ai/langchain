import pytest

from langchain.document_loaders import blob_loaders


def test_deprecated_error() -> None:
    """Hard-code public API to help determine if we have broken it."""
    deprecated = [
        "Blob",
        "BlobLoader",
        "FileSystemBlobLoader",
        "YoutubeAudioLoader",
    ]
    for import_ in deprecated:
        with pytest.raises(ImportError) as e:
            getattr(blob_loaders, import_)
            assert "langchain_community" in e.msg
