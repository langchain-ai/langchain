import json
import os
from io import BytesIO

import dotenv  # type: ignore[import-not-found]

from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

dotenv.load_dotenv()

POOL_MANAGEMENT_ENDPOINT = os.getenv("AZURE_DYNAMIC_SESSIONS_POOL_MANAGEMENT_ENDPOINT")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "testdata.txt")
TEST_DATA_CONTENT = open(TEST_DATA_PATH, "rb").read()


def test_end_to_end() -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)  # type: ignore[arg-type]
    result = tool.run("print('hello world')\n1 + 1")
    assert json.loads(result) == {
        "result": 2,
        "stdout": "hello world\n",
        "stderr": "",
    }

    # upload file content
    uploaded_file1_metadata = tool.upload_file(
        remote_file_path="test1.txt", data=BytesIO(b"hello world!!!!!")
    )
    assert uploaded_file1_metadata.filename == "test1.txt"
    assert uploaded_file1_metadata.size_in_bytes == 16
    assert uploaded_file1_metadata.full_path == "/mnt/data/test1.txt"
    downloaded_file1 = tool.download_file(remote_file_path="test1.txt")
    assert downloaded_file1.read() == b"hello world!!!!!"

    # upload file from buffer
    with open(TEST_DATA_PATH, "rb") as f:
        uploaded_file2_metadata = tool.upload_file(remote_file_path="test2.txt", data=f)
        assert uploaded_file2_metadata.filename == "test2.txt"
        downloaded_file2 = tool.download_file(remote_file_path="test2.txt")
        assert downloaded_file2.read() == TEST_DATA_CONTENT

    # upload file from disk, specifying remote file path
    uploaded_file3_metadata = tool.upload_file(
        remote_file_path="test3.txt", local_file_path=TEST_DATA_PATH
    )
    assert uploaded_file3_metadata.filename == "test3.txt"
    downloaded_file3 = tool.download_file(remote_file_path="test3.txt")
    assert downloaded_file3.read() == TEST_DATA_CONTENT

    # upload file from disk, without specifying remote file path
    uploaded_file4_metadata = tool.upload_file(local_file_path=TEST_DATA_PATH)
    assert uploaded_file4_metadata.filename == os.path.basename(TEST_DATA_PATH)
    downloaded_file4 = tool.download_file(
        remote_file_path=uploaded_file4_metadata.filename
    )
    assert downloaded_file4.read() == TEST_DATA_CONTENT

    # list files
    remote_files_metadata = tool.list_files()
    assert len(remote_files_metadata) == 4
    remote_file_paths = [metadata.filename for metadata in remote_files_metadata]
    expected_filenames = [
        "test1.txt",
        "test2.txt",
        "test3.txt",
        os.path.basename(TEST_DATA_PATH),
    ]
    assert set(remote_file_paths) == set(expected_filenames)
