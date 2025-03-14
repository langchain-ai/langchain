from langchain_community.document_loaders import BiliBiliLoader


def test_bilibili_loader() -> None:
    """Test Bilibili Loader."""
    loader = BiliBiliLoader(
        [
            "https://www.bilibili.com/video/BV1xt411o7Xu/",
            "https://www.bilibili.com/video/av330407025/",
            "https://www.bilibili.com/video/BV16b4y1R7wP/?p=5",
        ]
    )
    docs = loader.load()
    assert len(docs) == 3
    assert docs[0].metadata["aid"] == 34218168
    assert docs[1].metadata["videos"] == 1
    assert docs[2].metadata["pages"][5 - 1]["cid"] == 300059803
    assert docs[2].metadata["cid"] == 300048569
