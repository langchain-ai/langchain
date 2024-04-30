from unittest.mock import Mock

from langchain_milvus.vectorstores import Zilliz


def test_initialization() -> None:
    """Test integration zilliz initialization."""
    embedding = Mock()
    Zilliz(
        embedding_function=embedding,
        connection_args={
            "uri": "",
            "user": "",
            "password": "",
            "secure": True,
        },
    )
