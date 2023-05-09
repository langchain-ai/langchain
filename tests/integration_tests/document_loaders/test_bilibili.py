from langchain.document_loaders.bilibili import BiliBiliLoader


def test_bilibili_loader() -> None:
    """Test Bilibili Loader."""
    loader = BiliBiliLoader(
        [
            "https://www.bilibili.com/video/BV1xt411o7Xu/",
            "https://www.bilibili.com/video/av330407025/",
        ]
    )
    docs = loader.load()

    assert len(docs) == 2

    assert len(docs[0].page_content) > 0
    assert docs[1].metadata["owner"]["mid"] == 398095160

    assert docs[1].page_content == ""
    assert docs[1].metadata["owner"]["mid"] == 398095160
