"""Integration tests for the deepagents sandbox backend abstraction.

Implementers should subclass this test suite and provide a fixture that returns a
clean `SandboxBackendProtocol` instance.

Example:
```python
from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from my_pkg import make_sandbox


class TestMySandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        backend = make_sandbox()
        try:
            yield backend
        finally:
            backend.delete()
```

"""

# ruff: noqa: E402, S108

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import pytest

deepagents = pytest.importorskip("deepagents")

from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)

from langchain_tests.base import BaseStandardTests

if TYPE_CHECKING:
    from collections.abc import Iterator


class SandboxIntegrationTests(BaseStandardTests):
    """Standard integration tests for a `SandboxBackendProtocol` implementation."""

    @pytest.fixture(scope="class")
    def sandbox_backend(
        self, sandbox: SandboxBackendProtocol
    ) -> SandboxBackendProtocol:
        """Provide the sandbox backend under test.

        Resets the shared test directory before yielding.
        """
        sandbox.execute(
            "rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops"
        )
        return sandbox

    @abstractmethod
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        """Yield a clean sandbox backend and tear it down after the class."""

    @property
    def has_sync(self) -> bool:
        """Whether the sandbox supports sync methods."""
        return True

    @property
    def has_async(self) -> bool:
        """Whether the sandbox supports async methods."""
        return True

    @pytest.fixture(autouse=True)
    def _setup_test_dir(self, sandbox_backend: SandboxBackendProtocol) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.execute(
            "rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops"
        )

    def test_write_new_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Write a new file and verify it can be read back via command execution."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/new_file.txt"
        content = "Hello, sandbox!\nLine 2\nLine 3"
        result = sandbox_backend.write(test_path, content)
        assert result.error is None
        assert result.path == test_path
        exec_result = sandbox_backend.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_read_basic_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Write a file and verify `read()` returns expected contents."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/read_test.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.read(test_path)
        assert "Error:" not in result
        assert all(line in result for line in ("Line 1", "Line 2", "Line 3"))

    def test_edit_single_occurrence(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Edit a file and assert exactly one occurrence was replaced."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = "/tmp/test_sandbox_ops/edit_single.txt"
        content = "Hello world\nGoodbye world\nHello again"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.edit(test_path, "Goodbye", "Farewell")
        assert result.error is None
        assert result.occurrences == 1
        file_content = sandbox_backend.read(test_path)
        assert "Farewell world" in file_content
        assert "Goodbye" not in file_content

    def test_ls_info_lists_files(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Create files and verify `ls_info()` lists them."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/a.txt", "a")
        sandbox_backend.write("/tmp/test_sandbox_ops/b.txt", "b")
        info = sandbox_backend.ls_info("/tmp/test_sandbox_ops")
        paths = sorted([i["path"] for i in info])
        assert "/tmp/test_sandbox_ops/a.txt" in paths
        assert "/tmp/test_sandbox_ops/b.txt" in paths

    def test_glob_info(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Create files and verify `glob_info()` returns expected matches."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/x.py", "print('x')")
        sandbox_backend.write("/tmp/test_sandbox_ops/y.txt", "y")
        matches = sandbox_backend.glob_info("*.py", path="/tmp/test_sandbox_ops")
        assert [m["path"] for m in matches] == ["x.py"]

    def test_grep_raw_literal(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Verify `grep_raw()` performs literal matching on special characters."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write("/tmp/test_sandbox_ops/grep.txt", "a (b)\nstr | int\n")
        matches = sandbox_backend.grep_raw("str | int", path="/tmp/test_sandbox_ops")
        assert isinstance(matches, list)
        assert matches[0]["path"].endswith("/grep.txt")
        assert matches[0]["text"].strip() == "str | int"

    def test_upload_single_file(self, sandbox_backend: SandboxBackendProtocol) -> None:
        """Upload one file and verify its contents on the sandbox."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_upload_single.txt"
        test_content = b"Hello, Sandbox!"

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].path == test_path
        assert upload_responses[0].error is None

        result = sandbox_backend.execute(f"cat {test_path}")
        assert result.output.strip() == test_content.decode()

    def test_download_single_file(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Upload then download a file and verify bytes match."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_download_single.txt"
        test_content = b"Download test content"

        sandbox_backend.upload_files([(test_path, test_content)])

        download_responses = sandbox_backend.download_files([test_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == test_path
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None

    def test_upload_download_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Upload then download and verify bytes survive a roundtrip."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_roundtrip.txt"
        test_content = b"Roundtrip test: special chars \n\t\r\x00"

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [
            FileDownloadResponse(path=test_path, content=test_content, error=None)
        ]

    def test_upload_multiple_files_order_preserved(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Uploading multiple files should preserve input order in responses."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            ("/tmp/test_multi_1.txt", b"Content 1"),
            ("/tmp/test_multi_2.txt", b"Content 2"),
            ("/tmp/test_multi_3.txt", b"Content 3"),
        ]

        upload_responses = sandbox_backend.upload_files(files)

        assert upload_responses == [
            FileUploadResponse(path=files[0][0], error=None),
            FileUploadResponse(path=files[1][0], error=None),
            FileUploadResponse(path=files[2][0], error=None),
        ]

    def test_download_multiple_files_order_preserved(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Downloading multiple files should preserve input order in responses."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            ("/tmp/test_batch_1.txt", b"Batch 1"),
            ("/tmp/test_batch_2.txt", b"Batch 2"),
            ("/tmp/test_batch_3.txt", b"Batch 3"),
        ]
        sandbox_backend.upload_files(files)

        paths = [p for p, _ in files]
        download_responses = sandbox_backend.download_files(paths)

        assert download_responses == [
            FileDownloadResponse(path=files[0][0], content=files[0][1], error=None),
            FileDownloadResponse(path=files[1][0], content=files[1][1], error=None),
            FileDownloadResponse(path=files[2][0], content=files[2][1], error=None),
        ]

    def test_upload_binary_content_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Upload and download binary bytes (0..255) without corruption."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/binary_file.bin"
        test_content = bytes(range(256))

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [
            FileDownloadResponse(path=test_path, content=test_content, error=None)
        ]

    def test_download_error_file_not_found(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Downloading a missing file should return `error="file_not_found"`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        missing_path = "/tmp/nonexistent_test_file.txt"

        responses = sandbox_backend.download_files([missing_path])

        assert responses == [
            FileDownloadResponse(
                path=missing_path, content=None, error="file_not_found"
            )
        ]

    def test_download_error_is_directory(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Downloading a directory should fail with a reasonable error code."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = "/tmp/test_directory"
        sandbox_backend.execute(f"rm -rf {dir_path} && mkdir -p {dir_path}")

        responses = sandbox_backend.download_files([dir_path])

        assert len(responses) == 1
        assert responses[0].path == dir_path
        assert responses[0].content is None
        assert responses[0].error in {"is_directory", "file_not_found", "invalid_path"}

    def test_download_error_permission_denied(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Downloading a chmod 000 file should fail with a reasonable error code."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = "/tmp/test_no_read.txt"
        sandbox_backend.execute(
            f"rm -f {test_path} && echo secret > {test_path} && chmod 000 {test_path}"
        )

        try:
            responses = sandbox_backend.download_files([test_path])
        finally:
            sandbox_backend.execute(f"chmod 644 {test_path} || true")

        assert len(responses) == 1
        assert responses[0].path == test_path
        assert responses[0].content is None
        assert responses[0].error in {
            "permission_denied",
            "file_not_found",
            "invalid_path",
        }

    def test_download_error_invalid_path_relative(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Downloading a relative path should fail with `error="invalid_path"`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        responses = sandbox_backend.download_files(["relative/path.txt"])

        assert responses == [
            FileDownloadResponse(
                path="relative/path.txt",
                content=None,
                error="invalid_path",
            )
        ]

    def test_upload_missing_parent_dir_or_roundtrip(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Uploading into a missing parent dir should error or roundtrip.

        Some sandboxes auto-create parent directories; others return an error.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = "/tmp/test_upload_missing_parent_dir"
        path = f"{dir_path}/deepagents_test_upload.txt"
        content = b"nope"
        sandbox_backend.execute(f"rm -rf {dir_path}")

        responses = sandbox_backend.upload_files([(path, content)])
        assert len(responses) == 1
        assert responses[0].path == path

        if responses[0].error is not None:
            assert responses[0].error in {
                "invalid_path",
                "permission_denied",
                "file_not_found",
            }
            return

        download = sandbox_backend.download_files([path])
        assert download == [
            FileDownloadResponse(path=path, content=content, error=None)
        ]

    def test_upload_relative_path_returns_invalid_path(
        self,
        sandbox_backend: SandboxBackendProtocol,
    ) -> None:
        """Uploading to a relative path should fail with `error="invalid_path"`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        path = "relative_upload.txt"
        content = b"nope"
        responses = sandbox_backend.upload_files([(path, content)])

        assert responses == [FileUploadResponse(path=path, error="invalid_path")]
