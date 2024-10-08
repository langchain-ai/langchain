from langchain_community.document_loaders.alibaba_oss_file_loader import (
    AlibabaOSSFileLoader,
)

BUCKET = ""
FILE_KEY = ""
ENDPOINT = ""
ACCESS_KEY_ID = ""
ACCESS_KEY_SECRET = ""


def test_oss_file_loader() -> None:
    """Test Alibaba Cloud OSS file loader."""
    loader = AlibabaOSSFileLoader(
        BUCKET, FILE_KEY, ENDPOINT, ACCESS_KEY_ID, ACCESS_KEY_SECRET
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content is not None
