from langchain_community.document_loaders import BiliBiliLoader


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
    assert docs[0].metadata["aid"] == 34218168
    assert docs[1].metadata["videos"] == 1
