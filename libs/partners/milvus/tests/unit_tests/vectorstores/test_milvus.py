import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from langchain_milvus.vectorstores import Milvus


def test_initialization() -> None:
    """Test integration milvus initialization."""
    embedding = Mock()
    with TemporaryDirectory() as tmp_dir:
        Milvus(
            embedding_function=embedding,
            connection_args={
                "uri": os.path.join(tmp_dir, "milvus.db"),
            },
        )
