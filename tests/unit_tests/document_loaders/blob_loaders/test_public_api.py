from langchain.document_loaders.blob_loaders import __all__


def test_public_api() -> None:
    """Hard-code public API to help determine if we have broken it."""
    assert sorted(__all__) == [
        "Blob",
        "BlobLoader",
        "FileSystemBlobLoader",
        "YoutubeAudioLoader",
    ]
