import os
from langchain_azure import SessionsPythonREPLTool
import dotenv


dotenv.load_dotenv()

POOL_MANAGEMENT_ENDPOINT = os.getenv("POOL_MANAGEMENT_ENDPOINT")


def test_end_to_end():
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    result = tool.run("print('hello world')\n1 + 1")
    assert result == "result:\n2\n\nstdout:\nhello world\n\n\nstderr:\n"

    uploaded_file_metadata = tool.upload_file(data=b"hello world!!!!!", remote_filename="test.txt")
    assert uploaded_file_metadata.filename == "test.txt"
    assert uploaded_file_metadata.size_in_bytes == 16
    assert uploaded_file_metadata.full_path == "/mnt/data/test.txt"

    remote_files_metadata = tool.list_files()
    assert len(remote_files_metadata) == 1
    assert remote_files_metadata[0].filename == "test.txt"

    downloaded_file = tool.download_file(remote_filename="test.txt")
    assert downloaded_file.read() == b"hello world!!!!!"