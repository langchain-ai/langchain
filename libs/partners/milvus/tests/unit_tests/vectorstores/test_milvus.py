from unittest.mock import Mock

from langchain_milvus.vectorstores import Milvus


def test_initialization() -> None:
    """Test integration milvus initialization."""
    embedding = Mock()
    Milvus(
        embedding_function=embedding,
        connection_args={
            "uri": "http://127.0.0.1:19530",
            "user": "",
            "password": "",
            "secure": False,
        },
    )
