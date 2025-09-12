"""Test UnDatasIO loader."""
from unittest.mock import MagicMock, patch

from langchain.document_loaders import UnDatasIOLoader


@patch("langchain.document_loaders.undatasio.UnDatasIO")  # ①  patch 路径要对
def test_undatasio_loader(MockClient):
    """Basic smoke test."""
    instance = MockClient.return_value
    instance.workspace_list.return_value = [{"work_id": "w1"}]
    instance.task_list.return_value = [{"task_id": "t1"}]
    instance.get_task_files.return_value = [
        {"file_id": "f1", "file_name": "test.pdf", "status": "parser success"}
    ]
    instance.upload_file.return_value = True
    instance.parse_files.return_value = True
    instance.get_parse_result.return_value = ["hello", "world"]

    loader = UnDatasIOLoader(token="fake", file_path="test.pdf")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "hello\nworld"
