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

import asyncio
import base64
import shlex
import sys
from abc import abstractmethod
from typing import TYPE_CHECKING

import pytest

deepagents = pytest.importorskip("deepagents")

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    ReadResult,
    SandboxBackendProtocol,
)

from langchain_tests.base import BaseStandardTests

if TYPE_CHECKING:
    from collections.abc import Iterator


def _quote(path: str) -> str:
    return shlex.quote(path)


class SandboxIntegrationTests(BaseStandardTests):
    """Standard integration tests for a `SandboxBackendProtocol` implementation."""

    @property
    def sandbox_root_dir(self) -> str:
        """Base directory used by sandbox file-operation tests."""
        return "/tmp/test_sandbox_ops/"

    def sandbox_path(self, relative_path: str, *, root_dir: str | None = None) -> str:
        """Build a path under the configured sandbox test directory."""
        root = root_dir or self.sandbox_root_dir
        return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"

    @pytest.fixture(scope="class")
    def sandbox_backend(
        self, sandbox: SandboxBackendProtocol
    ) -> SandboxBackendProtocol:
        """Provide the sandbox backend under test.

        Resets the shared test directory before yielding.
        """
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
    def sandbox_test_root(self, request: pytest.FixtureRequest) -> str:
        """Create an isolated sandbox root directory for each test case."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        node_name = request.node.name.replace("/", "_").replace(" ", "_")
        return self.sandbox_path(node_name)

    def test_write_new_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Write a new file and verify it can be read back via command execution."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = self.sandbox_path("new_file.txt", root_dir=sandbox_test_root)
        content = "Hello, sandbox!\nLine 2\nLine 3"
        result = sandbox_backend.write(test_path, content)
        assert result.error is None
        assert result.path == test_path
        exec_result = sandbox_backend.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_read_basic_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Write a file and verify `read()` returns expected contents."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = self.sandbox_path("read_test.txt", root_dir=sandbox_test_root)
        content = "Line 1\nLine 2\nLine 3"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.read(test_path)
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert all(
            line in result.file_data["content"]
            for line in ("Line 1", "Line 2", "Line 3")
        )

    def test_read_binary_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload a binary file and verify `read()` returns base64-encoded content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = self.sandbox_path("binary.png", root_dir=sandbox_test_root)
        raw_bytes = bytes(range(256))
        sandbox_backend.upload_files([(test_path, raw_bytes)])
        result = sandbox_backend.read(test_path)
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    def test_read_binary_file_100_kib(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Read should return base64 content for a 100 KiB binary file."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("binary_100kib.png", root_dir=sandbox_test_root)
        chunk = bytes(range(256))
        raw_bytes = chunk * 400

        sandbox_backend.upload_files([(test_path, raw_bytes)])
        result = sandbox_backend.read(test_path)

        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    def test_read_binary_file_1_mib_returns_error(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Read should error when a binary file exceeds the preview size limit."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("binary_1mib.png", root_dir=sandbox_test_root)
        chunk = bytes(range(256))
        raw_bytes = chunk * 4096

        sandbox_backend.upload_files([(test_path, raw_bytes)])
        result = sandbox_backend.read(test_path)

        assert isinstance(result, ReadResult)
        assert result.file_data is None
        expected_error = (
            f"File '{test_path}': Binary file exceeds maximum preview size of "
            "512000 bytes"
        )
        assert result.error == expected_error

    def test_execute_large_stdout_payload(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Execute should handle a command that emits about 500 KiB of stdout."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        command = "python -c \"import sys; sys.stdout.write('x' * (500 * 1024))\""
        result = sandbox_backend.execute(command)

        assert result.exit_code == 0
        assert result.truncated is False
        assert len(result.output) >= 500 * 1024
        assert result.output.startswith("x")

    def test_edit_single_occurrence(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Edit a file and assert exactly one occurrence was replaced."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        test_path = self.sandbox_path("edit_single.txt", root_dir=sandbox_test_root)
        content = "Hello world\nGoodbye world\nHello again"
        sandbox_backend.write(test_path, content)
        result = sandbox_backend.edit(test_path, "Goodbye", "Farewell")
        assert result.error is None
        assert result.occurrences == 1
        file_result = sandbox_backend.read(test_path)
        assert isinstance(file_result, ReadResult)
        assert file_result.error is None
        assert file_result.file_data is not None
        assert "Farewell world" in file_result.file_data["content"]
        assert "Goodbye" not in file_result.file_data["content"]

    def test_ls_lists_files(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Create files and verify `ls()` lists them."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write(
            self.sandbox_path("a.txt", root_dir=sandbox_test_root), "a"
        )
        sandbox_backend.write(
            self.sandbox_path("b.txt", root_dir=sandbox_test_root), "b"
        )
        result = sandbox_backend.ls(sandbox_test_root)
        assert result.error is None
        assert result.entries is not None
        paths = sorted([i["path"] for i in result.entries])
        assert self.sandbox_path("a.txt", root_dir=sandbox_test_root) in paths
        assert self.sandbox_path("b.txt", root_dir=sandbox_test_root) in paths

    def test_glob(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Create files and verify `glob()` returns expected matches."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write(
            self.sandbox_path("x.py", root_dir=sandbox_test_root), "print('x')"
        )
        sandbox_backend.write(
            self.sandbox_path("y.txt", root_dir=sandbox_test_root), "y"
        )
        result = sandbox_backend.glob("*.py", path=sandbox_test_root)
        assert result.error is None
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["x.py"]

    def test_grep_literal(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Verify `grep()` performs literal matching on special characters."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")
        sandbox_backend.write(
            self.sandbox_path("grep.txt", root_dir=sandbox_test_root),
            "a (b)\nstr | int\n",
        )
        result = sandbox_backend.grep("str | int", path=sandbox_test_root)
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) > 0
        assert result.matches[0]["path"].endswith("/grep.txt")
        assert result.matches[0]["text"].strip() == "str | int"

    def test_upload_single_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload one file and verify its contents on the sandbox."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "test_upload_single.txt", root_dir=sandbox_test_root
        )
        test_content = b"Hello, Sandbox!"

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])

        assert len(upload_responses) == 1
        assert upload_responses[0].path == test_path
        assert upload_responses[0].error is None

        result = sandbox_backend.execute(f"cat {test_path}")
        assert result.output.strip() == test_content.decode()

    def test_download_single_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload then download a file and verify bytes match."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "test_download_single.txt", root_dir=sandbox_test_root
        )
        test_content = b"Download test content"

        sandbox_backend.upload_files([(test_path, test_content)])

        download_responses = sandbox_backend.download_files([test_path])

        assert len(download_responses) == 1
        assert download_responses[0].path == test_path
        assert download_responses[0].content == test_content
        assert download_responses[0].error is None

    def test_upload_download_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload then download and verify bytes survive a roundtrip."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("test_roundtrip.txt", root_dir=sandbox_test_root)
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
        sandbox_test_root: str,
    ) -> None:
        """Uploading multiple files should preserve input order in responses."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            (
                self.sandbox_path("test_multi_1.txt", root_dir=sandbox_test_root),
                b"Content 1",
            ),
            (
                self.sandbox_path("test_multi_2.txt", root_dir=sandbox_test_root),
                b"Content 2",
            ),
            (
                self.sandbox_path("test_multi_3.txt", root_dir=sandbox_test_root),
                b"Content 3",
            ),
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
        sandbox_test_root: str,
    ) -> None:
        """Downloading multiple files should preserve input order in responses."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        files = [
            (
                self.sandbox_path("test_batch_1.txt", root_dir=sandbox_test_root),
                b"Batch 1",
            ),
            (
                self.sandbox_path("test_batch_2.txt", root_dir=sandbox_test_root),
                b"Batch 2",
            ),
            (
                self.sandbox_path("test_batch_3.txt", root_dir=sandbox_test_root),
                b"Batch 3",
            ),
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
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload and download binary bytes (0..255) without corruption."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("binary_file.bin", root_dir=sandbox_test_root)
        test_content = bytes(range(256))

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [
            FileDownloadResponse(path=test_path, content=test_content, error=None)
        ]

    def test_upload_large_file_reports_expected_size(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Upload a ~10 MiB file, verify its size, then download it again."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("large_upload.txt", root_dir=sandbox_test_root)
        chunk = b"0123456789abcdef" * 1024
        repeat_count = 640
        test_content = chunk * repeat_count

        assert len(test_content) == 10 * 1024 * 1024

        upload_responses = sandbox_backend.upload_files([(test_path, test_content)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        exec_result = sandbox_backend.execute(f"wc -c {_quote(test_path)}")
        assert exec_result.exit_code == 0
        assert str(len(test_content)) in exec_result.output

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [
            FileDownloadResponse(path=test_path, content=test_content, error=None)
        ]

    def test_download_error_file_not_found(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Downloading a missing file should return `error="file_not_found"`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        missing_path = self.sandbox_path(
            "nonexistent_test_file.txt", root_dir=sandbox_test_root
        )

        responses = sandbox_backend.download_files([missing_path])

        assert responses == [
            FileDownloadResponse(
                path=missing_path, content=None, error="file_not_found"
            )
        ]

    def test_download_error_is_directory(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Downloading a directory should fail with a reasonable error code."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = self.sandbox_path("test_directory", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"rm -rf {dir_path} && mkdir -p {dir_path}")

        responses = sandbox_backend.download_files([dir_path])

        assert len(responses) == 1
        assert responses[0].path == dir_path
        assert responses[0].content is None
        assert responses[0].error in {"is_directory", "file_not_found", "invalid_path"}

    def test_download_error_permission_denied(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Downloading a chmod 000 file should fail with a reasonable error code."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("test_no_read.txt", root_dir=sandbox_test_root)
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
        sandbox_test_root: str,
    ) -> None:
        """Uploading into a missing parent dir should error or roundtrip.

        Some sandboxes auto-create parent directories; others return an error.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        dir_path = self.sandbox_path(
            "test_upload_missing_parent_dir", root_dir=sandbox_test_root
        )
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

    def test_write_creates_parent_dirs(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing into a missing nested directory should succeed."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "deep/nested/dir/file.txt", root_dir=sandbox_test_root
        )
        content = "Nested file content"

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        assert result.path == test_path
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == content

    def test_write_existing_file_fails(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing to an existing file should return an error without overwriting."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("existing.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "First content")

        result = sandbox_backend.write(test_path, "Second content")

        assert result.error is not None
        assert "already exists" in result.error.lower()
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == "First content"

    def test_write_special_characters(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing should preserve shell-sensitive characters exactly."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("special.txt", root_dir=sandbox_test_root)
        content = (
            "Special chars: $VAR, `command`, $(subshell), 'quotes', \"quotes\"\n"
            "Tab\there\n"
            "Backslash: \\\\"
        )

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == content

    def test_write_empty_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing empty content should still create the file."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("empty.txt", root_dir=sandbox_test_root)

        result = sandbox_backend.write(test_path, "")

        assert result.error is None
        exec_result = sandbox_backend.execute(
            f"[ -f {_quote(test_path)} ] && echo exists || echo missing"
        )
        assert "exists" in exec_result.output

    def test_write_path_with_spaces(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing should support file paths containing spaces."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "dir with spaces/file name.txt", root_dir=sandbox_test_root
        )
        content = "Content in file with spaces"

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == content

    def test_write_unicode_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing should preserve unicode content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("unicode.txt", root_dir=sandbox_test_root)
        content = "Hello 👋 世界 مرحبا Привет 🌍\nLine with émojis 🎉"

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == content

    def test_write_consecutive_slashes_in_path(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing should tolerate normalized paths with consecutive slashes."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("file.txt", root_dir=sandbox_test_root)
        content = "Content"

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        exec_result = sandbox_backend.execute(f"cat {_quote(test_path)}")
        assert exec_result.output.strip() == content

    def test_write_very_long_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing moderately long multi-line content should succeed."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("very_long.txt", root_dir=sandbox_test_root)
        content = "\n".join([f"Line {i} with some content here" for i in range(1000)])

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        read_result = sandbox_backend.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "Line 0 with some content here" in read_result.file_data["content"]

    def test_write_content_with_only_newlines(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Writing newline-only content should preserve the newline count."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("only_newlines.txt", root_dir=sandbox_test_root)
        content = "\n\n\n\n\n"

        result = sandbox_backend.write(test_path, content)

        assert result.error is None
        exec_result = sandbox_backend.execute(f"wc -l {_quote(test_path)}")
        assert "5" in exec_result.output

    def test_read_nonexistent_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading a missing file should return a file-not-found style error."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        result = sandbox_backend.read(
            self.sandbox_path("nonexistent.txt", root_dir=sandbox_test_root)
        )

        assert result.error is not None
        assert (
            "not_found" in result.error.lower() or "not found" in result.error.lower()
        )

    def test_read_empty_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading an empty file should succeed with empty-or-empty-notice content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("empty_read.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "")

        result = sandbox_backend.read(test_path)

        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "empty" in content.lower() or content.strip() == ""

    def test_read_with_offset(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading with offset should skip the requested number of lines."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("offset_test.txt", root_dir=sandbox_test_root)
        content = "\n".join([f"Row_{i}_content" for i in range(1, 11)])
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path, offset=5)

        assert result.error is None
        assert result.file_data is not None
        assert "Row_6_content" in result.file_data["content"]
        assert "Row_1_content" not in result.file_data["content"]

    def test_read_with_limit(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading with limit should cap the number of returned lines."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("limit_test.txt", root_dir=sandbox_test_root)
        content = "\n".join([f"Row_{i}_content" for i in range(1, 101)])
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path, offset=0, limit=5)

        assert result.error is None
        assert result.file_data is not None
        assert "Row_1_content" in result.file_data["content"]
        assert "Row_5_content" in result.file_data["content"]
        assert "Row_6_content" not in result.file_data["content"]

    def test_read_with_offset_and_limit(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading with offset and limit should return the expected slice."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "offset_limit_test.txt", root_dir=sandbox_test_root
        )
        content = "\n".join([f"Row_{i}_content" for i in range(1, 21)])
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path, offset=10, limit=5)

        assert result.error is None
        assert result.file_data is not None
        assert "Row_11_content" in result.file_data["content"]
        assert "Row_15_content" in result.file_data["content"]
        assert "Row_10_content" not in result.file_data["content"]
        assert "Row_16_content" not in result.file_data["content"]

    def test_read_unicode_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading unicode content should preserve non-ASCII text."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("unicode_read.txt", root_dir=sandbox_test_root)
        content = "Hello 👋 世界\nПривет мир\nمرحبا العالم"  # noqa: RUF001
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path)

        assert result.error is None
        assert result.file_data is not None
        assert "👋" in result.file_data["content"]
        assert "世界" in result.file_data["content"]
        assert "Привет" in result.file_data["content"]

    def test_read_file_with_very_long_lines(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading files with long lines should still succeed."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("long_lines.txt", root_dir=sandbox_test_root)
        long_line = "x" * 3000
        content = f"Short line\n{long_line}\nAnother short line"
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path)

        assert result.error is None
        assert result.file_data is not None
        assert "Short line" in result.file_data["content"]

    def test_read_with_zero_limit(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading with `limit=0` should not include file content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("zero_limit.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "Line 1\nLine 2\nLine 3")

        result = sandbox_backend.read(test_path, offset=0, limit=0)

        content = result.file_data["content"] if result.file_data else ""
        assert "Line 1" not in content or content.strip() == ""

    def test_read_offset_beyond_file_length(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading beyond EOF should return no file lines."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("offset_beyond.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "Line 1\nLine 2\nLine 3")

        result = sandbox_backend.read(test_path, offset=100, limit=10)

        content = result.file_data["content"] if result.file_data else ""
        error = result.error or ""
        assert "Line 1" not in content
        assert "Line 1" not in error
        assert "Line 2" not in content
        assert "Line 2" not in error
        assert "Line 3" not in content
        assert "Line 3" not in error

    def test_read_offset_at_exact_file_length(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Reading exactly at EOF should return no file lines."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("offset_exact.txt", root_dir=sandbox_test_root)
        content = "\n".join([f"Line {i}" for i in range(1, 6)])
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.read(test_path, offset=5, limit=10)

        text = result.file_data["content"] if result.file_data else ""
        error = result.error or ""
        assert "Line 1" not in text
        assert "Line 1" not in error
        assert "Line 5" not in text
        assert "Line 5" not in error

    def test_read_very_large_file_in_chunks(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Repeated offset+limit reads should cover different slices of a large file."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("large_chunked.txt", root_dir=sandbox_test_root)
        content = "\n".join([f"Line_{i:04d}_content" for i in range(1000)])
        sandbox_backend.write(test_path, content)

        first = sandbox_backend.read(test_path, offset=0, limit=100)
        middle = sandbox_backend.read(test_path, offset=500, limit=100)
        last = sandbox_backend.read(test_path, offset=900, limit=100)

        assert first.error is None
        assert first.file_data is not None
        assert "Line_0000_content" in first.file_data["content"]
        assert "Line_0099_content" in first.file_data["content"]
        assert "Line_0100_content" not in first.file_data["content"]

        assert middle.error is None
        assert middle.file_data is not None
        assert "Line_0500_content" in middle.file_data["content"]
        assert "Line_0599_content" in middle.file_data["content"]
        assert "Line_0499_content" not in middle.file_data["content"]

        assert last.error is None
        assert last.file_data is not None
        assert "Line_0900_content" in last.file_data["content"]
        assert "Line_0999_content" in last.file_data["content"]

    def test_edit_multiple_occurrences_without_replace_all(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing multiple matches without `replace_all` should fail."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("edit_multi.txt", root_dir=sandbox_test_root)
        content = "apple\nbanana\napple\norange\napple"
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.edit(test_path, "apple", "pear", replace_all=False)

        assert result.error is not None
        assert "multiple" in result.error.lower()
        read_result = sandbox_backend.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "apple" in read_result.file_data["content"]
        assert "pear" not in read_result.file_data["content"]

    def test_edit_multiple_occurrences_with_replace_all(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing multiple matches with `replace_all` should replace each match."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "edit_replace_all.txt", root_dir=sandbox_test_root
        )
        content = "apple\nbanana\napple\norange\napple"
        sandbox_backend.write(test_path, content)

        result = sandbox_backend.edit(test_path, "apple", "pear", replace_all=True)

        assert result.error is None
        assert result.occurrences == 3
        read_result = sandbox_backend.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "apple" not in read_result.file_data["content"]
        assert read_result.file_data["content"].count("pear") == 3

    def test_edit_string_not_found(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing a missing string should return a not-found style error."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("edit_not_found.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "Hello world")

        result = sandbox_backend.edit(test_path, "nonexistent", "replacement")

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_nonexistent_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing a missing file should return a file-not-found style error."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        result = sandbox_backend.edit(
            self.sandbox_path("nonexistent_edit.txt", root_dir=sandbox_test_root),
            "old",
            "new",
        )

        assert result.error is not None
        assert (
            "not_found" in result.error.lower() or "not found" in result.error.lower()
        )

    def test_edit_special_characters(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing should treat special characters as literal strings."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("edit_special.txt", root_dir=sandbox_test_root)
        content = "Price: $100.00\nPattern: [a-z]*\nPath: /usr/bin"
        sandbox_backend.write(test_path, content)

        first = sandbox_backend.edit(test_path, "$100.00", "$200.00")
        second = sandbox_backend.edit(test_path, "[a-z]*", "[0-9]+")

        assert first.error is None
        assert second.error is None
        read_result = sandbox_backend.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "$200.00" in read_result.file_data["content"]
        assert "[0-9]+" in read_result.file_data["content"]

    def test_edit_multiline_support(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Editing should support replacing multi-line strings."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path("edit_multiline.txt", root_dir=sandbox_test_root)
        sandbox_backend.write(test_path, "Line 1\nLine 2\nLine 3")

        result = sandbox_backend.edit(test_path, "Line 1\nLine 2", "Combined")

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox_backend.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "Combined" in read_result.file_data["content"]

    def test_ls_lists_nested_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Listing should include nested directories and immediate child files."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("ls_nested", root_dir=sandbox_test_root)
        sandbox_backend.execute(
            f"mkdir -p {_quote(base_dir)}/subdir && touch {_quote(base_dir)}/root.txt"
        )

        result = sandbox_backend.ls(base_dir)

        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert f"{base_dir}/subdir" in paths
        assert f"{base_dir}/root.txt" in paths

    def test_ls_unicode_filenames(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Listing should preserve unicode filenames."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("ls_unicode", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/测试文件.txt", "content")
        sandbox_backend.write(f"{base_dir}/файл.txt", "content")

        result = sandbox_backend.ls(base_dir)

        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert f"{base_dir}/测试文件.txt" in paths
        assert f"{base_dir}/файл.txt" in paths

    def test_ls_large_directory(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Listing a larger directory should include all created entries."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("ls_large", root_dir=sandbox_test_root)
        sandbox_backend.execute(
            f"mkdir -p {_quote(base_dir)} && "
            f"cd {_quote(base_dir)} && "
            "for i in $(seq 0 49); do "
            "echo content > file_$(printf '%03d' $i).txt; "
            "done"
        )

        result = sandbox_backend.ls(base_dir)

        assert result.error is None
        assert result.entries is not None
        assert len(result.entries) == 50
        paths = [entry["path"] for entry in result.entries]
        assert f"{base_dir}/file_000.txt" in paths
        assert f"{base_dir}/file_049.txt" in paths

    def test_ls_path_with_trailing_slash(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Listing a path with a trailing slash should match the normalized path."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("ls_trailing", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file.txt", "content")

        result = sandbox_backend.ls(f"{base_dir}/")

        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert f"{base_dir}/file.txt" in paths

    def test_ls_special_characters_in_filenames(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Listing should preserve filenames with shell metacharacters."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("ls_special", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file(1).txt", "content")
        sandbox_backend.write(f"{base_dir}/file[2].txt", "content")
        sandbox_backend.write(f"{base_dir}/file-3.txt", "content")

        result = sandbox_backend.ls(base_dir)

        assert result.error is None
        assert result.entries is not None
        paths = [entry["path"] for entry in result.entries]
        assert f"{base_dir}/file(1).txt" in paths
        assert f"{base_dir}/file[2].txt" in paths
        assert f"{base_dir}/file-3.txt" in paths

    def test_ls_path_is_sanitized(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Listing an injected path should not execute attacker-controlled code."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        malicious_path = "'; import os; os.system('echo INJECTED'); #"
        result = sandbox_backend.ls(malicious_path)

        assert result.error is not None or result.entries == []

    def test_read_path_is_sanitized(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Reading an injected path should return an error without executing it."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        malicious_path = "'; import os; os.system('echo INJECTED'); #"
        result = sandbox_backend.read(malicious_path)

        assert result.error is not None
        assert result.file_data is None

    def test_grep_basic_search(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should return matches across multiple files."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_test", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file1.txt", "Hello world\nGoodbye world")
        sandbox_backend.write(f"{base_dir}/file2.txt", "Hello there\nGoodbye friend")

        result = sandbox_backend.grep("Hello", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 2
        paths = [match["path"] for match in result.matches]
        assert any(path.endswith("file1.txt") for path in paths)
        assert any(path.endswith("file2.txt") for path in paths)
        assert all(match["line"] == 1 for match in result.matches)

    def test_grep_with_glob_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should honor the file glob filter."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_glob", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/test.txt", "pattern")
        sandbox_backend.write(f"{base_dir}/test.py", "pattern")
        sandbox_backend.write(f"{base_dir}/test.md", "pattern")

        result = sandbox_backend.grep("pattern", path=base_dir, glob="*.py")

        assert result.error is None
        assert result.matches == [
            {"path": f"{base_dir}/test.py", "line": 1, "text": "pattern"}
        ]

    def test_grep_no_matches(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep with no matches should return an empty match list."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_empty", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file.txt", "Hello world")

        result = sandbox_backend.grep("nonexistent", path=base_dir)

        assert result.error is None
        assert result.matches == []

    def test_grep_multiple_matches_per_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should report multiple matches from a single file with line numbers."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_multi", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(
            f"{base_dir}/fruits.txt", "apple\nbanana\napple\norange\napple"
        )

        result = sandbox_backend.grep("apple", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3
        assert [match["line"] for match in result.matches] == [1, 3, 5]

    def test_grep_literal_string_matching(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should treat the search pattern literally rather than as regex."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_literal", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/numbers.txt", "test123\ntest456\nabcdef")

        result = sandbox_backend.grep("test123", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "test123" in result.matches[0]["text"]

    def test_grep_unicode_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should match unicode patterns in unicode content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_unicode", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(
            f"{base_dir}/unicode.txt",
            "Hello 世界\nПривет мир\n测试 pattern",  # noqa: RUF001
        )

        result = sandbox_backend.grep("世界", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "世界" in result.matches[0]["text"]

    def test_grep_case_sensitivity(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should be case-sensitive by default."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_case", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/case.txt", "Hello\nhello\nHELLO")

        result = sandbox_backend.grep("Hello", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["text"] == "Hello"

    def test_grep_with_special_characters(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should treat special characters in the pattern literally."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_special", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(
            f"{base_dir}/special.txt", "Price: $100\nPath: /usr/bin\nPattern: [a-z]*"
        )

        dollar = sandbox_backend.grep("$100", path=base_dir)
        brackets = sandbox_backend.grep("[a-z]*", path=base_dir)

        assert dollar.error is None
        assert dollar.matches is not None
        assert len(dollar.matches) == 1
        assert "$100" in dollar.matches[0]["text"]

        assert brackets.error is None
        assert brackets.matches is not None
        assert len(brackets.matches) == 1
        assert "[a-z]*" in brackets.matches[0]["text"]

    def test_grep_empty_directory(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep in an empty directory should return no matches."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_empty_dir", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")

        result = sandbox_backend.grep("anything", path=base_dir)

        assert result.error is None
        assert result.matches == []

    def test_grep_across_nested_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should recurse into nested directories."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_nested", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}/sub1/sub2")
        sandbox_backend.write(f"{base_dir}/root.txt", "target here")
        sandbox_backend.write(f"{base_dir}/sub1/level1.txt", "target here")
        sandbox_backend.write(f"{base_dir}/sub1/sub2/level2.txt", "target here")

        result = sandbox_backend.grep("target", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3

    def test_grep_with_globstar_include_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep with a glob filter should still find nested matching files."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_globstar", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}/a/b")
        sandbox_backend.write(f"{base_dir}/a/b/target.py", "needle")
        sandbox_backend.write(f"{base_dir}/a/ignore.txt", "needle")

        result = sandbox_backend.grep("needle", path=base_dir, glob="*.py")

        assert result.error is None
        assert result.matches == [
            {"path": f"{base_dir}/a/b/target.py", "line": 1, "text": "needle"}
        ]

    def test_grep_reports_correct_line_numbers(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Grep should report the original file line number for a match."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("grep_multiline", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        sandbox_backend.write(f"{base_dir}/long.txt", content)

        result = sandbox_backend.grep("Line 50", path=base_dir)

        assert result.error is None
        assert result.matches == [
            {"path": f"{base_dir}/long.txt", "line": 50, "text": "Line 50"}
        ]

    def test_glob_basic_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should match basic wildcard patterns."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_test", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file1.txt", "content")
        sandbox_backend.write(f"{base_dir}/file2.txt", "content")
        sandbox_backend.write(f"{base_dir}/file3.py", "content")

        result = sandbox_backend.glob("*.txt", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "file1.txt" in paths
        assert "file2.txt" in paths
        assert not any(path.endswith(".py") for path in paths)

    def test_glob_recursive_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should support recursive patterns with `**`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_recursive", root_dir=sandbox_test_root)
        sandbox_backend.execute(
            f"mkdir -p {_quote(base_dir)}/subdir1 {_quote(base_dir)}/subdir2"
        )
        sandbox_backend.write(f"{base_dir}/root.txt", "content")
        sandbox_backend.write(f"{base_dir}/subdir1/nested1.txt", "content")
        sandbox_backend.write(f"{base_dir}/subdir2/nested2.txt", "content")

        result = sandbox_backend.glob("**/*.txt", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert any(path.endswith("nested1.txt") for path in paths)
        assert any(path.endswith("nested2.txt") for path in paths)

    def test_glob_no_matches(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob with no matches should return an empty match list."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_empty", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file.txt", "content")

        result = sandbox_backend.glob("*.py", path=base_dir)

        assert result.error is None
        assert result.matches == []

    def test_glob_with_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should include directories and mark them with `is_dir`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_dirs", root_dir=sandbox_test_root)
        sandbox_backend.execute(
            f"mkdir -p {_quote(base_dir)}/dir1 {_quote(base_dir)}/dir2"
        )
        sandbox_backend.write(f"{base_dir}/file.txt", "content")

        result = sandbox_backend.glob("*", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 3
        dir_count = sum(1 for info in result.matches if info["is_dir"])
        file_count = sum(1 for info in result.matches if not info["is_dir"])
        assert dir_count == 2
        assert file_count == 1

    def test_glob_hidden_files_explicitly(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should match hidden files when the pattern explicitly requests them."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_hidden", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/.hidden1", "content")
        sandbox_backend.write(f"{base_dir}/.hidden2", "content")
        sandbox_backend.write(f"{base_dir}/visible.txt", "content")

        result = sandbox_backend.glob(".*", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert ".hidden1" in paths or ".hidden2" in paths
        assert not any(path == "visible.txt" for path in paths)

    def test_glob_with_character_class(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should support character classes in patterns."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_charclass", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file1.txt", "content")
        sandbox_backend.write(f"{base_dir}/file2.txt", "content")
        sandbox_backend.write(f"{base_dir}/file3.txt", "content")
        sandbox_backend.write(f"{base_dir}/fileA.txt", "content")

        result = sandbox_backend.glob("file[1-2].txt", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "file1.txt" in paths
        assert "file2.txt" in paths
        assert "file3.txt" not in paths
        assert "fileA.txt" not in paths

    def test_glob_with_question_mark(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Glob should support single-character wildcards."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        base_dir = self.sandbox_path("glob_question", root_dir=sandbox_test_root)
        sandbox_backend.execute(f"mkdir -p {_quote(base_dir)}")
        sandbox_backend.write(f"{base_dir}/file1.txt", "content")
        sandbox_backend.write(f"{base_dir}/file2.txt", "content")
        sandbox_backend.write(f"{base_dir}/file10.txt", "content")

        result = sandbox_backend.glob("file?.txt", path=base_dir)

        assert result.error is None
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert len(paths) == 2
        assert "file1.txt" in paths
        assert "file2.txt" in paths
        assert "file10.txt" not in paths

    async def test_awrite_aread_large_text_payload(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async write should allow a large text file to be read back non-empty."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "large_async_text.txt", root_dir=sandbox_test_root
        )
        line = "0123456789abcdef" * 256
        lines = [line for _ in range(2560)]
        test_content = "\n".join(lines)

        write_result = await sandbox_backend.awrite(test_path, test_content)
        assert write_result.error is None
        assert write_result.path == test_path

        exec_result = await sandbox_backend.aexecute(f"wc -c {_quote(test_path)}")
        assert exec_result.exit_code == 0
        assert str(len(test_content.encode("utf-8"))) in exec_result.output

        read_result = await sandbox_backend.aread(test_path)
        assert isinstance(read_result, ReadResult)
        assert read_result.error is None
        assert read_result.file_data is not None
        assert read_result.file_data["encoding"] == "utf-8"
        assert read_result.file_data["content"].startswith(lines[0])

    async def test_aread_large_text_payload_paginated_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async paginated reads should reconstruct the full large text payload."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "large_async_chunked.txt", root_dir=sandbox_test_root
        )
        lines = [f"Line_{i:04d}_content" for i in range(2500)]
        test_content = "\n".join(lines)

        write_result = await sandbox_backend.awrite(test_path, test_content)
        assert write_result.error is None

        parts: list[str] = []
        for offset in range(0, len(lines), 100):
            page = await sandbox_backend.aread(test_path, offset=offset, limit=100)
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"] == "\n".join(lines[offset : offset + 100])
            parts.append(page.file_data["content"])

        assert "\n".join(parts) == test_content

    async def test_adownload_large_text_payload_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async download should preserve the full large text payload exactly."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "large_async_download.txt", root_dir=sandbox_test_root
        )
        line = "0123456789abcdef" * 256
        lines = [line for _ in range(2560)]
        test_content = "\n".join(lines)

        write_result = await sandbox_backend.awrite(test_path, test_content)
        assert write_result.error is None

        download_responses = await sandbox_backend.adownload_files([test_path])
        assert download_responses == [
            FileDownloadResponse(
                path=test_path,
                content=test_content.encode("utf-8"),
                error=None,
            )
        ]

    def test_write_read_download_large_text_with_escaped_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Sync large-text roundtrips should preserve escaped and unicode content."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        test_path = self.sandbox_path(
            "large_sync_escaped.txt", root_dir=sandbox_test_root
        )
        line = (
            "prefix\t\u2603\u4e16\u754c\u03c0\u22483.14159"
            " | spaces   preserved"
            " | quotes ' \""
            " | brackets [] {{}}"
            " | shell $VAR `cmd` $(subshell)"
            " | slash /tmp/path and backslash \\\\"
            " | control-ish \\r \\n"
            " | suffix"
        )
        lines = [f"{i:04d}:{line}" for i in range(2500)]
        test_content = "\n".join(lines)

        write_result = sandbox_backend.write(test_path, test_content)
        assert write_result.error is None

        pages: list[str] = []
        for offset in range(0, len(lines), 100):
            page = sandbox_backend.read(test_path, offset=offset, limit=100)
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"] == "\n".join(lines[offset : offset + 100])
            pages.append(page.file_data["content"])

        assert "\n".join(pages) == test_content

        download_responses = sandbox_backend.download_files([test_path])
        assert download_responses == [
            FileDownloadResponse(
                path=test_path,
                content=test_content.encode("utf-8"),
                error=None,
            )
        ]

    async def test_awrite_aread_adownload_large_text_with_escaped_content(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async large-text roundtrips should preserve escaped and unicode content."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "large_async_escaped.txt", root_dir=sandbox_test_root
        )
        line = (
            "prefix\t\u2603\u4e16\u754c\u03c0\u22483.14159"
            " | spaces   preserved"
            " | quotes ' \""
            " | brackets [] {{}}"
            " | shell $VAR `cmd` $(subshell)"
            " | slash /tmp/path and backslash \\\\"
            " | control-ish \\r \\n"
            " | suffix"
        )
        lines = [f"{i:04d}:{line}" for i in range(2500)]
        test_content = "\n".join(lines)

        write_result = await sandbox_backend.awrite(test_path, test_content)
        assert write_result.error is None

        pages: list[str] = []
        for offset in range(0, len(lines), 100):
            page = await sandbox_backend.aread(test_path, offset=offset, limit=100)
            assert page.error is None
            assert page.file_data is not None
            assert page.file_data["content"] == "\n".join(lines[offset : offset + 100])
            pages.append(page.file_data["content"])

        assert "\n".join(pages) == test_content

        download_responses = await sandbox_backend.adownload_files([test_path])
        assert download_responses == [
            FileDownloadResponse(
                path=test_path,
                content=test_content.encode("utf-8"),
                error=None,
            )
        ]

    async def test_aread_binary_image_file(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async read should return base64-encoded content for a binary image file."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path("async_binary.png", root_dir=sandbox_test_root)
        raw_bytes = bytes(range(256))

        upload_responses = await sandbox_backend.aupload_files([(test_path, raw_bytes)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        result = await sandbox_backend.aread(test_path)
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    async def test_aread_binary_file_100_kib(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async read should return base64 content for a 100 KiB binary file."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "async_binary_100kib.png", root_dir=sandbox_test_root
        )
        chunk = bytes(range(256))
        raw_bytes = chunk * 400

        upload_responses = await sandbox_backend.aupload_files([(test_path, raw_bytes)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        result = await sandbox_backend.aread(test_path)
        assert isinstance(result, ReadResult)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw_bytes

    async def test_aread_binary_file_1_mib_returns_error(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async read should error when a binary file exceeds the preview size limit."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "async_binary_1mib.png", root_dir=sandbox_test_root
        )
        chunk = bytes(range(256))
        raw_bytes = chunk * 4096

        upload_responses = await sandbox_backend.aupload_files([(test_path, raw_bytes)])
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        result = await sandbox_backend.aread(test_path)
        assert isinstance(result, ReadResult)
        assert result.file_data is None
        expected_error = (
            f"File '{test_path}': Binary file exceeds maximum preview size of "
            "512000 bytes"
        )
        assert result.error == expected_error

    async def test_aexecute_large_stdout_payload(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        """Async execute should handle five parallel 500 KiB stdout commands."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        command = "python -c \"import sys; sys.stdout.write('x' * (500 * 1024))\""
        if sys.version_info >= (3, 11):
            tasks: list[asyncio.Task[ExecuteResponse]] = []
            async with asyncio.TaskGroup() as tg:
                tasks.extend(
                    tg.create_task(sandbox_backend.aexecute(command)) for _ in range(5)
                )

            for task in tasks:
                result = task.result()
                assert result.exit_code == 0
                assert result.truncated is False
                assert len(result.output) >= 500 * 1024
                assert result.output.startswith("x")
        else:
            pytest.skip("asyncio.TaskGroup requires Python 3.11+")

    async def test_aupload_adownload_large_file_roundtrip(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        """Async upload/download should preserve a ~10 MiB payload exactly."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        test_path = self.sandbox_path(
            "large_async_upload.bin", root_dir=sandbox_test_root
        )
        chunk = b"0123456789abcdef" * 1024
        repeat_count = 640
        test_content = chunk * repeat_count

        assert len(test_content) == 10 * 1024 * 1024

        upload_responses = await sandbox_backend.aupload_files(
            [(test_path, test_content)]
        )
        assert upload_responses == [FileUploadResponse(path=test_path, error=None)]

        exec_result = await sandbox_backend.aexecute(f"wc -c {_quote(test_path)}")
        assert exec_result.exit_code == 0
        assert str(len(test_content)) in exec_result.output

        download_responses = await sandbox_backend.adownload_files([test_path])
        assert download_responses == [
            FileDownloadResponse(path=test_path, content=test_content, error=None)
        ]
